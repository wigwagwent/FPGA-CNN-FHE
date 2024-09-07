use mnist_lib::{self, MnistImage};
use model_layers::layers::{self, dense_layer, flatten_layer};
use model_layers::{Activation, Quantized, SGDOptimizer, VecD1, VecD2, Weights};

mod model_layers;

fn main() {
    let images: Vec<MnistImage> = mnist_lib::load_mnist_dataset();

    let mut model = Model::random_weights();
    let optimizer = SGDOptimizer::new(0.01);

    // Training loop
    for epoch in 0..20 {
        let mut total_loss = 0.0;
        for (_idx, image) in images.iter().enumerate() {
            let image_data: VecD2 = image
                .data
                .iter()
                .map(|row| row.iter().map(|&pixel| pixel as f32 / 255.0).collect())
                .collect();

            let mut target = vec![0.0; 10];
            target[image.label.as_usize()] = 1.0;

            let loss = model.train(image_data, target, &optimizer);
            total_loss += loss;
        }
        println!(
            "Epoch {} complete. Average loss: {}",
            epoch,
            total_loss / images.len() as f32
        );
        let mut correct_count = 0;
        for image in images.iter() {
            let image_data: VecD2 = image
                .data
                .iter()
                .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
                .collect();

            let output = model.forward(image_data);
            let predicted_class = get_predicted_class(output);

            if image.label == mnist_lib::MnistDigit::from_usize(predicted_class) {
                correct_count += 1;
            }
        }

        println!(
            "Accuracy after epoch: {:.2}%",
            correct_count as f32 / images.len() as f32 * 100.0
        );
    }

    // Evaluate the model after training
    let mut correct_count = 0;
    for image in images.iter() {
        let image_data: VecD2 = image
            .data
            .iter()
            .map(|row| row.iter().map(|&pixel| pixel as f32).collect())
            .collect();

        let output = model.forward(image_data);
        let predicted_class = get_predicted_class(output);

        if image.label == mnist_lib::MnistDigit::from_usize(predicted_class) {
            correct_count += 1;
        }
    }

    println!(
        "Accuracy after training: {:.2}%",
        correct_count as f32 / images.len() as f32 * 100.0
    );
}

// Structure to represent the machine learing model
pub struct Model {
    conv_weights: Vec<Weights>,
    dense_weights: Vec<Weights>,
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
            layers::convolution_layer(input, self.conv_weights.clone(), Activation::ReLU(0.0));
        let flatten_output = layers::flatten_layer(conv_output);
        layers::dense_layer(
            flatten_output,
            self.dense_weights.clone(),
            Activation::Softmax,
        )
    }

    pub fn train(&mut self, input: VecD2, target: VecD1, optimizer: &SGDOptimizer) -> Quantized {
        // Forward pass
        let conv_output = layers::convolution_layer(
            input.clone(),
            self.conv_weights.clone(),
            Activation::ReLU(0.0),
        );
        let flatten_output = flatten_layer(conv_output.clone());
        let dense_output = dense_layer(
            flatten_output.clone(),
            self.dense_weights.clone(),
            Activation::Softmax,
        );

        // Compute loss
        let loss = layers::cross_entropy_loss(&dense_output, &target);

        // Backward pass
        let (dense_gradients, flatten_grad) =
            layers::backprop_dense(&flatten_output, &dense_output, &target, &self.dense_weights);
        let conv_grad = layers::unflatten_gradient(&flatten_grad, &conv_output);
        let conv_gradients =
            layers::backprop_conv(&input, &conv_output, &conv_grad, &self.conv_weights);

        // Update weights
        for (weight, gradient) in self.dense_weights.iter_mut().zip(dense_gradients.iter()) {
            optimizer.update(weight, gradient);
        }
        for (weight, gradient) in self.conv_weights.iter_mut().zip(conv_gradients.iter()) {
            optimizer.update(weight, gradient);
        }

        loss
    }
}

fn get_predicted_class(probabilities: VecD1) -> usize {
    probabilities
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(index, _)| index)
        .unwrap()
}
