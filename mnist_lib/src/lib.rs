// use std::fs::File;
// use std::io::Read;

fn read_mnist_image_file(buffer: Vec<u8>) -> Vec<MnistImageData> {
    // Extract Image data
    assert!(buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 8 && buffer[3] == 3); // Magic number

    let num_images = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;
    let num_rows = u32::from_be_bytes(buffer[8..12].try_into().unwrap()) as usize;
    let num_cols = u32::from_be_bytes(buffer[12..16].try_into().unwrap()) as usize;

    assert!(num_rows == 28 && num_cols == 28); // Image dimensions

    let mut offset = 16;
    let mut mnist_images: Vec<MnistImageData> = Vec::new();

    for _ in 0..num_images {
        let mut image_data: Vec<Vec<u8>> = Vec::new();
        for _ in 0..28 {
            let mut row = Vec::new();
            for _ in 0..28 {
                row.push(buffer[offset]);
                offset += 1;
            }
            image_data.push(row);
        }
        mnist_images.push(image_data);
    }

    mnist_images
}

fn read_mnist_label_file(buffer: Vec<u8>) -> Vec<MnistDigit> {
    assert!(buffer[0] == 0 && buffer[1] == 0 && buffer[2] == 8 && buffer[3] == 1); // Magic number

    let num_labels = u32::from_be_bytes(buffer[4..8].try_into().unwrap()) as usize;

    let mut offset = 8;
    let mut mnist_labels: Vec<MnistDigit> = Vec::new();

    for _ in 0..num_labels {
        let label = MnistDigit::from_usize(buffer[offset] as usize);
        mnist_labels.push(label);
        offset += 1;
    }

    mnist_labels
}

pub fn load_mnist_dataset(dataset: MnistDataset) -> Vec<MnistImage> {
    let (image_buffer, label_buffer) = match dataset {
        MnistDataset::Train => (
            include_bytes!("../data/train-images.idx3-ubyte").to_vec(),
            include_bytes!("../data/train-labels.idx1-ubyte").to_vec(),
        ),
        MnistDataset::Validate => (
            include_bytes!("../data/t10k-images.idx3-ubyte").to_vec(),
            include_bytes!("../data/t10k-labels.idx1-ubyte").to_vec(),
        ),
    };

    let image_data = read_mnist_image_file(image_buffer);
    let label_data = read_mnist_label_file(label_buffer);

    assert!(image_data.len() == label_data.len()); // Check if Image and Label data have same number of elements

    // Combine Image and Label data
    image_data
        .iter()
        .zip(label_data.iter())
        .map(|(image, label)| MnistImage {
            data: image.clone(),
            label: *label,
        })
        .collect()
}

pub enum MnistDataset {
    Train,
    Validate,
}

pub type MnistImageData = Vec<Vec<u8>>;

#[derive(Debug, Clone, Copy, PartialEq)]
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
    use image::ImageBuffer;
    use std::fs::File;
    use std::io::Write;

    use super::*;

    #[test]
    fn train_image_0_and_label() {
        let data = load_mnist_dataset(MnistDataset::Train);

        let image_0 = data[0].data.clone();
        let label_0 = data[0].label;

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(28, 28, image_0.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/train_image_0.jpg").unwrap();

        let mut label_file = File::create("tests/train_label_0.txt").unwrap();
        label_file
            .write_all(label_0.as_usize().to_string().as_bytes())
            .unwrap();
    }

    #[test]
    fn train_image_100_and_label() {
        let data = load_mnist_dataset(MnistDataset::Train);

        let image_100 = data[100].data.clone();
        let label_100 = data[100].label;

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(28, 28, image_100.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/train_image_100.jpg").unwrap();

        let mut label_file = File::create("tests/train_label_100.txt").unwrap();
        label_file
            .write_all(label_100.as_usize().to_string().as_bytes())
            .unwrap();
    }

    #[test]
    fn validate_image_0_and_label() {
        let data = load_mnist_dataset(MnistDataset::Validate);

        let image_0 = data[0].data.clone();
        let label_0 = data[0].label;

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(28, 28, image_0.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/validate_image_0.jpg").unwrap();

        let mut label_file = File::create("tests/validate_label_0.txt").unwrap();
        label_file
            .write_all(label_0.as_usize().to_string().as_bytes())
            .unwrap();
    }

    #[test]
    fn validate_image_100_and_label() {
        let data = load_mnist_dataset(MnistDataset::Validate);

        let image_100 = data[100].data.clone();
        let label_100 = data[100].label;

        let image_buffer: ImageBuffer<image::Luma<u8>, Vec<u8>> =
            ImageBuffer::from_raw(28, 28, image_100.iter().flatten().cloned().collect()).unwrap();
        image_buffer.save("tests/validate_image_100.jpg").unwrap();

        let mut label_file = File::create("tests/validate_label_100.txt").unwrap();
        label_file
            .write_all(label_100.as_usize().to_string().as_bytes())
            .unwrap();
    }
}
