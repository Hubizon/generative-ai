const datasetSelect = document.getElementById("datasetSelect");
const modelSelect = document.getElementById("modelSelect");
const batchSizeInput = document.getElementById("batchSizeInput");
const useSeedCheckbox = document.getElementById("useSeedCheckbox");
const seedInput = document.getElementById("seedInput");
const generateButton = document.getElementById("generateButton");
const grid = document.getElementById("grid");

generateButton.addEventListener("click", async () => {
    // Clear previous images displayed in the grid and display a new one
    grid.innerHTML = "";
    await generateImages();
});

useSeedCheckbox.addEventListener("change", () => {
    seedInput.disabled = !useSeedCheckbox.checked; // Disable seed input if checkbox is unchecked
    if (!useSeedCheckbox.checked) seedInput.value = ""; // Clear seed input if checkbox is unchecked
});

// Asynchronous function to generate images
async function generateImages() {
    const Z_DIM = 100; // Latent vector size
    const X_DIM = 56 // Image size
    const dataset = datasetSelect.value; // Get selected dataset option
    const model = modelSelect.value; // Get selected model option
    const batchSize = Math.min(parseInt(batchSizeInput.value, 10) || 1, 45); // Limit batch size to 45
    const seed = parseInt(seedInput.value, 10); // Get the seed value from input

    try {
        // Generate latent vector for input to model
        const latentVector = generateLatentVector(batchSize * Z_DIM, seed);

        // Set the path to the model based on the selected dataset
        let modelPath = "Models/";
        if (dataset === "mnist")
            modelPath += "GAN_MNIST.onnx"; // Model for MNIST dataset
        else if (dataset === "fashion_mnist")
            modelPath += "GAN_Fashion_MNIST.onnx"; // Model for Fashion MNIST dataset

        // Create ONNX InferenceSession for the selected model
        const session = await ort.InferenceSession.create(modelPath);

        // Create tensor from latent vector for input to the model
        const inputTensor = new ort.Tensor("float32", latentVector, [batchSize, Z_DIM, 1, 1]);

        // Run the model with the latent vector as input
        const results = await session.run({ latent_vector: inputTensor });
        const generatedImages = results.generated_image.data;

        // Render each generated image on the page
        const imageSize = X_DIM * X_DIM;
        for (let i = 0; i < batchSize; i++) {
            const imageStart = i * imageSize;
            const imageEnd = imageStart + imageSize;
            const imageSlice = generatedImages.slice(imageStart, imageEnd);
            renderImage(imageSlice, X_DIM);
        }
    } catch (error) {
        console.error("Error during generation:", error); // Log errors if any occur
    }
}

// Function to generate a latent vector based on the seed or randomly
function generateLatentVector(size, seed) {
    if (useSeedCheckbox.checked) {
        // Initialize random number generator with the seed
        const validatedSeed = seed || 0; // default to 0
        const rng = new Math.seedrandom(validatedSeed);
        return new Float32Array(size).map(() => rng() * 2 - 1);
    } else {
        // Generate random vector if no seed
        return new Float32Array(size).map(() => Math.random() * 2 - 1);
    }
}

// Function to render the generated image on the page
function renderImage(imageData, x_dim) {
    // Create a 56x56 canvas element
    const canvas = document.createElement("canvas");
    canvas.width = x_dim;
    canvas.height = x_dim;
    const ctx = canvas.getContext("2d");
    const imageDataObj = ctx.createImageData(x_dim, x_dim);

    // Map generated image data to canvas pixel values
    for (let i = 0; i < imageData.length; i++) {
        const value = ((imageData[i] + 1) / 2) * 255;
        imageDataObj.data[i * 4] = value; // Red channel
        imageDataObj.data[i * 4 + 1] = value; // Green channel
        imageDataObj.data[i * 4 + 2] = value; // Blue channel
        imageDataObj.data[i * 4 + 3] = 255; // Alpha channel
    }

    ctx.putImageData(imageDataObj, 0, 0); // Put the image data on the canvas
    grid.appendChild(canvas); // Append the canvas to the grid to display the image
}
