use fhe_ckks::{Ciphertext, DoubleSized, Plaintext};
use mnist_lib::{self, MnistDataset, MnistImage};
use model_layers::{layers, WeightsFhe};
use model_layers::{Activation, Quantized, SGDOptimizer, VecD1, VecD2, Weights};
use num_traits::{PrimInt, Signed};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

mod model_layers;

fn main() {
    let val_images: Vec<MnistImage> = mnist_lib::load_mnist_dataset(MnistDataset::Validate);
    let train_images: Vec<MnistImage> = mnist_lib::load_mnist_dataset(MnistDataset::Train);

    let mut model = Model::random_weights();
    let mut optimizer = SGDOptimizer::new(0.01);
    optimizer.initialize_velocities(&model.dense_weights, &model.conv_weights);

    // Training loop
    for epoch in 0..20 {
        let epoch_start = Instant::now();
        let mut total_loss = 0.0;

        for (_idx, image) in train_images.iter().enumerate() {
            let image_data: VecD2 = image
                .data
                .iter()
                .map(|row| row.iter().map(|&pixel| pixel as f32 / 255.0).collect())
                .collect();

            let mut target = vec![0.0; 10];
            target[image.label.as_usize()] = 1.0;

            let loss = model.train(image_data, target, &mut optimizer);
            total_loss += loss;
        }

        let epoch_duration = epoch_start.elapsed();

        println!(
            "Epoch {} complete. Average loss: {:.6}. Duration: {:.2?}",
            epoch,
            total_loss / train_images.len() as f32,
            epoch_duration
        );

        // Evaluate the model after each epoch
        model.validate(&val_images);
        model.validate(&train_images);
    }
}

// Structure to represent the machine learing model
pub struct Model {
    conv_weights: Vec<Weights>,
    dense_weights: Vec<Weights>,
}

pub struct ModelFhe<T, const N: usize>
where
    T: PrimInt + Signed + DoubleSized,
{
    conv_weights: Vec<WeightsFhe<T, N>>,
    dense_weights: Vec<WeightsFhe<T, N>>,
}

impl<T, const N: usize> From<&Model> for ModelFhe<T, N>
where
    T: PrimInt + Signed + DoubleSized,
{
    fn from(model: &Model) -> Self {
        ModelFhe {
            conv_weights: model
                .conv_weights
                .iter()
                .map(|w| WeightsFhe::from(w.clone()))
                .collect(),
            dense_weights: model
                .dense_weights
                .iter()
                .map(|w| WeightsFhe::from(w.clone()))
                .collect(),
        }
    }
}

impl Model {
    pub fn new(mut weights: Vec<Weights>) -> Self {
        let dense_weights = weights.split_off(weights.len() - 10);
        let conv_weights = weights.split_off(weights.len() - 8);
        Model {
            conv_weights,
            dense_weights,
        }
    }

    pub fn random_weights() -> Self {
        Model {
            conv_weights: vec![Weights::rand_convolution(3); 8],
            dense_weights: vec![Weights::rand_dense(5408); 10],
        }
    }

    pub fn forward(&self, input: VecD2) -> VecD1 {
        let conv_output =
            layers::convolution_layer(input, self.conv_weights.clone(), Activation::Quadratic);
        let flatten_output = layers::flatten_layer(conv_output);
        layers::dense_layer(
            flatten_output,
            self.dense_weights.clone(),
            Activation::Softmax,
        )
    }

    pub fn forward_fhe(&self, input: VecD2) -> VecD1 {
        let fhe_model: ModelFhe<i64, 5408> = ModelFhe::from(self);

        let conv_output =
            layers::convolution_layer(input, self.conv_weights.clone(), Activation::Quadratic);

        // let conv_output_fhe: Vec<Vec<Ciphertext<i32, 15>>> = conv_output
        //     .iter()
        //     .map(|row| {
        //         row.iter()
        //             .map(|val| Plaintext::from_f32(val.clone(), 15).encrypt())
        //             .collect()
        //     })
        //     .collect();

        let flatten_output = layers::flatten_layer(conv_output);

        let flatten_output_fhe: Ciphertext<i64, 5408> =
            Plaintext::from_f32(flatten_output, 15).encrypt();

        let dense_output = layers::dense_layer_fhe(
            flatten_output_fhe,
            fhe_model.dense_weights.clone(),
            Activation::None,
        );

        let decrypted_dense_output: VecD1 = dense_output
            .iter()
            .map(|val| Vec::from(val.decrypt()).iter().sum())
            .collect();

        layers::activation_layer(decrypted_dense_output, Activation::Softmax)
    }

    pub fn train(
        &mut self,
        input: VecD2,
        target: VecD1,
        optimizer: &mut SGDOptimizer,
    ) -> Quantized {
        //self.forward_fhe(input.clone());
        // Forward pass
        let conv_output = layers::convolution_layer_par(
            input.clone(),
            self.conv_weights.clone(),
            Activation::Quadratic,
        );
        let flatten_output = layers::flatten_layer(conv_output.clone());
        let dense_output = layers::dense_layer_par(
            flatten_output.clone(),
            self.dense_weights.clone(),
            Activation::Softmax,
        );

        // Compute loss
        let loss = layers::cross_entropy_loss(&dense_output, &target);

        // Backpropagation
        let clip_value: Quantized = 2.5;
        let (dense_gradients, flatten_grad) = layers::backprop_dense(
            &flatten_output,
            &dense_output,
            &target,
            &self.dense_weights,
            clip_value,
        );
        let conv_grad = layers::unflatten_gradient(&flatten_grad, &conv_output);
        let conv_gradients = layers::backprop_conv(
            &input,
            &conv_output,
            &conv_grad,
            &self.conv_weights,
            clip_value,
        );

        // Update weights
        for (weight, gradient) in self.dense_weights.iter_mut().zip(dense_gradients.iter()) {
            optimizer.update(weight, gradient);
        }
        for (weight, gradient) in self.conv_weights.iter_mut().zip(conv_gradients.iter()) {
            optimizer.update(weight, gradient);
        }

        loss
    }

    fn validate(&self, images: &Vec<MnistImage>) -> usize {
        let correct_count = AtomicUsize::new(0);
        images.par_iter().for_each(|image| {
            let image_data: VecD2 = image
                .data
                .iter()
                .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
                .collect();

            let output = self.forward_fhe(image_data);
            let predicted_class = get_predicted_class(output);

            if image.label == mnist_lib::MnistDigit::from_usize(predicted_class) {
                correct_count.fetch_add(1, Ordering::Relaxed);
            }
        });

        let correct_count = correct_count.load(Ordering::Relaxed);

        println!(
            "Accuracy after training: {:.2}%",
            correct_count as f32 / images.len() as f32 * 100.0
        );
        correct_count
    }
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Less))
        .map(|(index, _)| index)
        .unwrap()
}
