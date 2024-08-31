use std::fs::File;
use std::io::{self, Read, Cursor};
use byteorder::{BigEndian, ReadBytesExt}; // Use byteorder crate for reading bytes

// Function to read weights from a binary file
fn read_weights(filename: &str) -> io::Result<Vec<Vec<i16>>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let num_filters = 32; // Adjust this to match the actual number of filters
    let kernel_size = 3 * 3; // Assuming a 3x3 kernel
    let weight_size = kernel_size * std::mem::size_of::<i16>();

    if buffer.len() < weight_size * num_filters {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "Buffer is too small"));
    }

    let mut weights = Vec::new();
    let mut offset = 0;

    for _ in 0..num_filters {
        let filter: Vec<i16> = buffer[offset..offset + weight_size]
            .chunks_exact(2)
            .map(|bytes| i16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();
        weights.push(filter);
        offset += weight_size;
    }
    
    Ok(weights)
}

// Function to read IDX images and return images
fn read_idx_images(filename: &str) -> io::Result<Vec<Vec<u8>>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut cursor = Cursor::new(buffer);

    // Read magic number
    let magic_number = cursor.read_u32::<BigEndian>()?;
    if magic_number != 2051 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid image file format"));
    }

    // Read dimensions
    let num_images = cursor.read_u32::<BigEndian>()?;
    let num_rows = cursor.read_u32::<BigEndian>()?;
    let num_cols = cursor.read_u32::<BigEndian>()?;
    
    let mut images = Vec::with_capacity(num_images as usize);
    let image_size = (num_rows * num_cols) as usize;
    for _ in 0..num_images {
        let mut image = vec![0; image_size];
        cursor.read_exact(&mut image)?;
        images.push(image);
    }

    Ok(images)
}

// Function to read IDX labels and return labels
fn read_idx_labels(filename: &str) -> io::Result<Vec<u8>> {
    let mut file = File::open(filename)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let mut cursor = Cursor::new(buffer);

    // Read magic number
    let magic_number = cursor.read_u32::<BigEndian>()?;
    if magic_number != 2049 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid label file format"));
    }

    // Read number of labels
    let num_labels = cursor.read_u32::<BigEndian>()?;
    let mut labels = vec![0; num_labels as usize];
    cursor.read_exact(&mut labels)?;

    Ok(labels)
}

// Integer-based convolution function
fn integer_convolution(input: &[Vec<i16>], kernel: &[Vec<i16>]) -> Vec<Vec<i16>> {
    let (input_height, input_width) = (input.len(), input[0].len());
    let (kernel_height, kernel_width) = (kernel.len(), kernel[0].len());

    // Check if kernel fits into input dimensions
    if input_height < kernel_height || input_width < kernel_width {
        panic!("Input dimensions are smaller than kernel dimensions");
    }

    let output_height = input_height.checked_sub(kernel_height)
        .expect("Input height is smaller than kernel height") + 1;
    let output_width = input_width.checked_sub(kernel_width)
        .expect("Input width is smaller than kernel width") + 1;

    let mut output = vec![vec![0; output_width]; output_height];
    
    for i in 0..output_height {
        for j in 0..output_width {
            let mut sum = 0;
            for ki in 0..kernel_height {
                for kj in 0..kernel_width {
                    sum += input[i + ki][j + kj] * kernel[ki][kj];
                }
            }
            output[i][j] = sum;
        }
    }

    output
}

// Flatten the output and perform a simple classification
fn classify_output(output: &[Vec<i16>]) -> usize {
    // Flatten the output and find the index with the maximum value
    let mut flat_output: Vec<i16> = output.iter().flat_map(|row| row.iter()).cloned().collect();
    let max_index = flat_output.iter().position(|&x| x == *flat_output.iter().max().unwrap()).unwrap();
    max_index
}

// Dummy fully connected layer for simplicity
fn fully_connected_layer(output: &[Vec<i16>], num_classes: usize) -> Vec<f32> {
    let mut scores = vec![0.0; num_classes];
    // Here we just use a dummy transformation, in practice you'd apply weights and biases
    for row in output {
        for &value in row {
            let index = (value as usize % num_classes).clamp(0, num_classes - 1);
            scores[index] += value as f32;
        }
    }
    scores
}

// Main function
fn main() -> io::Result<()> {
    // Load quantized weights
    let weights = read_weights("weights.bin")?;

    // Load MNIST dataset
    let images = read_idx_images("t10k-images.idx3-ubyte")?;
    let labels = read_idx_labels("t10k-labels.idx1-ubyte")?;

    // Convert images from u8 to i16
    let images_i16: Vec<Vec<i16>> = images.into_iter()
        .map(|image| image.into_iter().map(|x| x as i16).collect())
        .collect();

    // Create a dummy kernel (example with 3x3 kernel, adjust as needed)
    let kernel = vec![
        vec![1, 0, -1],
        vec![1, 0, -1],
        vec![1, 0, -1],
    ];

    let mut correct_count = 0;
    let mut incorrect_count = 0;

    // Iterate through all images and labels
    for (image, actual_label) in images_i16.iter().zip(&labels) {
        // Verify input image dimensions
        if image.len() != 28 * 28 {
            panic!("Input image dimensions do not match expected size of 28x28");
        }

        // Reshape input image to 2D (28x28)
        let reshaped_image: Vec<Vec<i16>> = image.chunks(28).map(|chunk| chunk.to_vec()).collect();

        // Perform inference on the reshaped image
        let conv_output = integer_convolution(&reshaped_image, &kernel);

        // Flatten the output and perform a simple classification
        let scores = fully_connected_layer(&conv_output, 10); // Assuming 10 classes for digits 0-9

        // Find the predicted digit
        let predicted_digit = scores.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).map(|(i, _)| i).unwrap_or(0);

        // Compare with the actual label
        if predicted_digit == *actual_label as usize {
            correct_count += 1;
        } else {
            incorrect_count += 1;
        }
    }

    println!("Total images: {}", labels.len());
    println!("Correct predictions: {}", correct_count);
    println!("Incorrect predictions: {}", incorrect_count);
    println!("Accuracy: {:.2}%", correct_count as f32 / labels.len() as f32 * 100.0);

    Ok(())
}
