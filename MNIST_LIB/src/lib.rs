use std::fs::File;
use std::io::Read;


fn read_mnist_image_file(file_path: &str) -> Vec<MnistImageData> {
    // Read Image data
    let mut image_file = File::open(file_path).expect(format!("{} file not found", file_path).as_str());
    let mut image_buffer = Vec::new();
    image_file.read_to_end(&mut image_buffer).unwrap();

    // Extract Image data
    assert!(image_buffer[0] == 0 && image_buffer[1] == 0 && image_buffer[2] == 8 && image_buffer[3] == 3); // Magic number

    let num_images = u32::from_be_bytes(image_buffer[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(image_buffer[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(image_buffer[12..16].try_into().unwrap()) as usize;

    assert!(num_rows == 28 && num_cols == 28); // Image dimensions

    let mut offset = 16;
    let mut mnist_images: Vec<MnistImageData> = Vec::new();

    for _ in 0..num_images {
        let mut image_data = [[0; 28]; 28];
        for i in 0..28 {
            for j in 0..28 {
                image_data[i][j] = image_buffer[offset];
                offset += 1;
            }
        }
        mnist_images.push(image_data);

    }

    mnist_images
}

fn read_mnist_label_file(file_path: &str) -> Vec<MnistDigit> {
    // Read Label data
    let mut label_file = File::open(file_path).expect(format!("{} file not found", file_path).as_str());
    let mut label_buffer = Vec::new();
    label_file.read_to_end(&mut label_buffer).unwrap();

    // Extract Label data
    assert!(label_buffer[0] == 0 && label_buffer[1] == 0 && label_buffer[2] == 8 && label_buffer[3] == 1); // Magic number

    let num_labels = u32::from_be_bytes(label_buffer[4..8].try_into().unwrap()) as usize;

    let mut offset = 8;
    let mut mnist_labels: Vec<MnistDigit> = Vec::new();

    for _ in 0..num_labels {
        let label = MnistDigit::from_usize(label_buffer[offset] as usize);
        mnist_labels.push(label);
        offset += 1;
    }

    mnist_labels
}


pub fn load_mnist_dataset() -> Vec<MnistImage> {
    // Read Image and Label data
    let image_data = read_mnist_image_file("data/t10k-images.idx3-ubyte");
    let label_data = read_mnist_label_file("data/t10k-labels.idx1-ubyte");

    assert!(image_data.len() == label_data.len()); // Check if Image and Label data have same number of elements

    // Combine Image and Label data
    image_data.iter().zip(label_data.iter()).map(|(image, label)| MnistImage { data: *image, label: *label }).collect()
}

pub type MnistImageData = [[u8; 28]; 28];

#[derive(Debug, Clone, Copy)]
pub enum MnistDigit {
    Zero,
    One,
    Two,
    Three,
    Four,
    Five,
    Six,
    Seven,
    Eight,
    Nine,
}

impl MnistDigit {
    pub fn as_usize(&self) -> usize {
        match self {
            MnistDigit::Zero => 0,
            MnistDigit::One => 1,
            MnistDigit::Two => 2,
            MnistDigit::Three => 3,
            MnistDigit::Four => 4,
            MnistDigit::Five => 5,
            MnistDigit::Six => 6,
            MnistDigit::Seven => 7,
            MnistDigit::Eight => 8,
            MnistDigit::Nine => 9,
        }
    }

    pub fn from_usize(digit: usize) -> MnistDigit {
        match digit {
            0 => MnistDigit::Zero,
            1 => MnistDigit::One,
            2 => MnistDigit::Two,
            3 => MnistDigit::Three,
            4 => MnistDigit::Four,
            5 => MnistDigit::Five,
            6 => MnistDigit::Six,
            7 => MnistDigit::Seven,
            8 => MnistDigit::Eight,
            9 => MnistDigit::Nine,
            _ => panic!("Invalid digit"),
        }
    }
}

pub struct MnistImage {
    pub data: MnistImageData,
    pub label: MnistDigit,
}




#[cfg(test)]
mod tests {
    use std::io::Write;
use image::ImageBuffer;

    use super::*;

    #[test]
    fn generate_and_save_image_0_and_label() {
        let image_data = read_mnist_image_file("data/t10k-images.idx3-ubyte");
        let label_data = read_mnist_label_file("data/t10k-labels.idx1-ubyte");

        let image_0 = image_data[0];
        let label_0 = label_data[0];

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(28, 28, image_0.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/image_0.jpg").unwrap();

        let mut label_file = File::create("tests/label_0.txt").unwrap();
        label_file.write_all(label_0.as_usize().to_string().as_bytes()).unwrap();
    }

    #[test]
    fn generate_and_save_image_100_and_label() {
        let image_data = read_mnist_image_file("data/t10k-images.idx3-ubyte");
        let label_data = read_mnist_label_file("data/t10k-labels.idx1-ubyte");

        let image_100 = image_data[100];
        let label_100 = label_data[100];

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(28, 28, image_100.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/image_100.jpg").unwrap();

        let mut label_file = File::create("tests/label_100.txt").unwrap();
        label_file.write_all(label_100.as_usize().to_string().as_bytes()).unwrap();
    }
}