use super::{Quantized, VecD1, VecD2};

pub fn relu_activation(mut input: VecD2, min: Quantized) -> VecD2 {
    for i in 0..input.len() {
        for j in 0..input[0].len() {
            input[i][j] = input[i][j].max(min);
        }
    }
    input
}

pub fn quadratic_activation(mut input: VecD2) -> VecD2 {
    for i in 0..input.len() {
        for j in 0..input[0].len() {
            input[i][j] = input[i][j] * input[i][j];
        }
    }
    input
}

pub fn softmax_activation(scores: VecD1) -> VecD1 {
    // Rest of the function remains the same
    let max_score = scores
        .iter()
        .max_by(|a, b| a.partial_cmp(b).expect("Cannot compare scores"))
        .expect("Scores vector is empty");

    let exp_scores: VecD1 = scores
        .iter()
        .map(|&score| (score - max_score).exp())
        .collect();
    let sum_exp_scores: Quantized = exp_scores.iter().sum();
    exp_scores
        .iter()
        .map(|&exp_score| exp_score / sum_exp_scores)
        .collect()
}
