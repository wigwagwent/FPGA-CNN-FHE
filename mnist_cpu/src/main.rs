use mnist_lib::{self, MnistDataset, MnistImage};
use model_layers::layers;
use model_layers::{Activation, Quantized, SGDOptimizer, VecD1, VecD2, Weights};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSlice;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

mod model_layers;

fn main() {
    let batch_size = 32;
    let num_epochs = 100;

    let val_images: Vec<MnistImage> = mnist_lib::load_mnist_dataset(MnistDataset::Validate);
    let train_images: Vec<MnistImage> = mnist_lib::load_mnist_dataset(MnistDataset::Train);

    let mut model = Model::random_weights();
    let mut optimizer = SGDOptimizer::new(0.01, batch_size); // Adjusted back to 0.01
    optimizer.initialize_velocities(&model.dense_weights, &model.conv_weights);

    for epoch in 0..num_epochs {
        let epoch_start = Instant::now();

        let (total_loss, mut total_gradients) = train_images
            .par_chunks(batch_size)
            .map(|batch| {
                let mut batch_loss = 0.0;
                let mut batch_gradients = model.create_zero_gradients();

                for image in batch {
                    let image_data: VecD2 = image
                        .data
                        .iter()
                        .map(|row| row.iter().map(|&pixel| pixel as f32 / 255.0).collect())
                        .collect();

                    let mut target = vec![0.0; 10];
                    target[image.label.as_usize()] = 1.0;

                    let (loss, gradients) = model.compute_gradients(image_data, target);
                    batch_loss += loss;
                    model.accumulate_gradients(&mut batch_gradients, &gradients);
                }

                // Average the gradients over the batch
                model.scale_gradients(&mut batch_gradients, 1.0 / batch.len() as f32);

                (batch_loss, batch_gradients)
            })
            .reduce(
                || (0.0, model.create_zero_gradients()),
                |(loss1, grad1), (loss2, grad2)| {
                    let total_loss = loss1 + loss2;
                    let total_gradients = model.sum_gradients(&grad1, &grad2);
                    (total_loss, total_gradients)
                },
            );

        // Scale the total gradients by the number of batches
        let num_batches = (train_images.len() + batch_size - 1) / batch_size;
        model.scale_gradients(&mut total_gradients, 1.0 / num_batches as f32);

        // Update model parameters using the accumulated gradients
        model.update_parameters(total_gradients, &mut optimizer);

        let epoch_duration = epoch_start.elapsed();
        println!(
            "Epoch {} complete. Average loss: {:.6}. Duration: {:.2?}",
            epoch,
            total_loss / train_images.len() as f32,
            epoch_duration
        );

        // Evaluate the model after each epoch
        if epoch % 10 == 0 {
            let val_accuracy = model.validate(&val_images);
            let train_accuracy = model.validate(&train_images);

            println!(
                "Validation accuracy: {:.2}%",
                val_accuracy as f32 / val_images.len() as f32 * 100.0
            );
            println!(
                "Training accuracy: {:.2}%",
                train_accuracy as f32 / train_images.len() as f32 * 100.0
            );
        }
    }
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

    pub fn train(
        &mut self,
        input: VecD2,
        target: VecD1,
        optimizer: &mut SGDOptimizer,
    ) -> Quantized {
        // Forward pass
        let conv_output = layers::convolution_layer(
            input.clone(),
            self.conv_weights.clone(),
            Activation::ReLU(0.0),
        );
        let flatten_output = layers::flatten_layer(conv_output.clone());
        let dense_output = layers::dense_layer(
            flatten_output.clone(),
            self.dense_weights.clone(),
            Activation::Softmax,
        );

        // Compute loss
        let loss = layers::cross_entropy_loss(&dense_output, &target);

        // Backpropagation
        let clip_value: Quantized = 1.0;
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

            let output = self.forward(image_data);
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

    pub fn scale_gradients(&self, gradients: &mut (Vec<Weights>, Vec<Weights>), scale: f32) {
        for grad in gradients.0.iter_mut().chain(gradients.1.iter_mut()) {
            match grad {
                Weights::Dense { weights, bias } => {
                    for w in weights.iter_mut() {
                        *w *= scale;
                    }
                    *bias *= scale;
                }
                Weights::Convolution { kernel, bias } => {
                    for row in kernel.iter_mut() {
                        for w in row.iter_mut() {
                            *w *= scale;
                        }
                    }
                    *bias *= scale;
                }
            }
        }
    }

    pub fn create_zero_gradients(&self) -> (Vec<Weights>, Vec<Weights>) {
        let dense_gradients = self
            .dense_weights
            .iter()
            .map(|w| match w {
                Weights::Dense { weights, .. } => Weights::Dense {
                    weights: vec![0.0; weights.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for dense layer"),
            })
            .collect();

        let conv_gradients = self
            .conv_weights
            .iter()
            .map(|w| match w {
                Weights::Convolution { kernel, .. } => Weights::Convolution {
                    kernel: vec![vec![0.0; kernel[0].len()]; kernel.len()],
                    bias: 0.0,
                },
                _ => panic!("Unexpected weight type for convolution layer"),
            })
            .collect();

        (dense_gradients, conv_gradients)
    }

    pub fn compute_gradients(
        &self,
        input: VecD2,
        target: Vec<f32>,
    ) -> (f32, (Vec<Weights>, Vec<Weights>)) {
        // Forward pass
        let conv_output = layers::convolution_layer(
            input.clone(),
            self.conv_weights.clone(),
            Activation::ReLU(0.0),
        );
        let flatten_output = layers::flatten_layer(conv_output.clone());
        let dense_output = layers::dense_layer(
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

        (loss, (dense_gradients, conv_gradients))
    }

    pub fn accumulate_gradients(
        &self,
        acc_gradients: &mut (Vec<Weights>, Vec<Weights>),
        gradients: &(Vec<Weights>, Vec<Weights>),
    ) {
        for (acc, grad) in acc_gradients.0.iter_mut().zip(gradients.0.iter()) {
            match (acc, grad) {
                (
                    Weights::Dense {
                        weights: acc_w,
                        bias: acc_b,
                    },
                    Weights::Dense {
                        weights: grad_w,
                        bias: grad_b,
                    },
                ) => {
                    for (a, g) in acc_w.iter_mut().zip(grad_w.iter()) {
                        *a += g;
                    }
                    *acc_b += grad_b;
                }
                _ => panic!("Mismatched weight types"),
            }
        }

        for (acc, grad) in acc_gradients.1.iter_mut().zip(gradients.1.iter()) {
            match (acc, grad) {
                (
                    Weights::Convolution {
                        kernel: acc_k,
                        bias: acc_b,
                    },
                    Weights::Convolution {
                        kernel: grad_k,
                        bias: grad_b,
                    },
                ) => {
                    for (a_row, g_row) in acc_k.iter_mut().zip(grad_k.iter()) {
                        for (a, g) in a_row.iter_mut().zip(g_row.iter()) {
                            *a += g;
                        }
                    }
                    *acc_b += grad_b;
                }
                _ => panic!("Mismatched weight types"),
            }
        }
    }

    pub fn sum_gradients(
        &self,
        grad1: &(Vec<Weights>, Vec<Weights>),
        grad2: &(Vec<Weights>, Vec<Weights>),
    ) -> (Vec<Weights>, Vec<Weights>) {
        let dense_sum = grad1
            .0
            .iter()
            .zip(grad2.0.iter())
            .map(|(w1, w2)| match (w1, w2) {
                (
                    Weights::Dense {
                        weights: w1,
                        bias: b1,
                    },
                    Weights::Dense {
                        weights: w2,
                        bias: b2,
                    },
                ) => Weights::Dense {
                    weights: w1.iter().zip(w2.iter()).map(|(a, b)| a + b).collect(),
                    bias: b1 + b2,
                },
                _ => panic!("Mismatched weight types"),
            })
            .collect();

        let conv_sum = grad1
            .1
            .iter()
            .zip(grad2.1.iter())
            .map(|(w1, w2)| match (w1, w2) {
                (
                    Weights::Convolution {
                        kernel: k1,
                        bias: b1,
                    },
                    Weights::Convolution {
                        kernel: k2,
                        bias: b2,
                    },
                ) => Weights::Convolution {
                    kernel: k1
                        .iter()
                        .zip(k2.iter())
                        .map(|(row1, row2)| {
                            row1.iter().zip(row2.iter()).map(|(a, b)| a + b).collect()
                        })
                        .collect(),
                    bias: b1 + b2,
                },
                _ => panic!("Mismatched weight types"),
            })
            .collect();

        (dense_sum, conv_sum)
    }

    pub fn update_parameters(
        &mut self,
        gradients: (Vec<Weights>, Vec<Weights>),
        optimizer: &mut SGDOptimizer,
    ) {
        for (weight, gradient) in self.dense_weights.iter_mut().zip(gradients.0.iter()) {
            optimizer.update(weight, gradient);
        }

        for (weight, gradient) in self.conv_weights.iter_mut().zip(gradients.1.iter()) {
            optimizer.update(weight, gradient);
        }
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
