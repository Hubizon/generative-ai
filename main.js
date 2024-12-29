const generateButton = document.getElementById("generateButton");
const useSeedCheckbox = document.getElementById("useSeedCheckbox");
const seedInput = document.getElementById("seedInput");
const grid = document.getElementById("grid");

// TODO: add an event listening to 'enter' (generate an image)
// TODO: add a checkbox to choose from AutoEncoder / GAN / Diffusion Model

generateButton.addEventListener("click", () => {
    grid.innerHTML = ""; // TODO: a button / a checkbox to reset the grid
    generateImages();
});

useSeedCheckbox.addEventListener("change", () => {
    seedInput.disabled = !useSeedCheckbox.checked;
    if (!useSeedCheckbox.checked) seedInput.value = "";
});

function generateLatentVector(seed, size) {
    if (useSeedCheckbox.checked) {
        const validatedSeed = seed || 0;
        const rng = new Math.seedrandom(validatedSeed);
        return new Float32Array(size).map(() => rng() * 2 - 1);
    } else {
        return new Float32Array(size).map(() => Math.random() * 2 - 1);
    }
}

async function generateImages() {
    const Z_DIM = 100;
    const seed = parseInt(seedInput.value, 10);

    try {
        const latentVector = generateLatentVector(seed, Z_DIM);
        const reshapedLatentVector = new Float32Array(latentVector);

        const session = await ort.InferenceSession.create("Models/GAN.onnx");

        const inputTensor = new ort.Tensor("float32", reshapedLatentVector, [1, Z_DIM, 1, 1]);

        const results = await session.run({ latent_vector: inputTensor });
        const generatedImage = results.generated_image.data;

        renderImage(generatedImage);
    } catch (error) {
        console.error("Error during generation:", error);
    }
}

function renderImage(imageData) {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 64;
    const ctx = canvas.getContext("2d");
    const imageDataObj = ctx.createImageData(64, 64);

    for (let i = 0; i < imageData.length; i++) {
        const value = ((imageData[i] + 1) / 2) * 255;
        imageDataObj.data[i * 4] = value;
        imageDataObj.data[i * 4 + 1] = value;
        imageDataObj.data[i * 4 + 2] = value;
        imageDataObj.data[i * 4 + 3] = 255;
    }

    ctx.putImageData(imageDataObj, 0, 0);
    grid.appendChild(canvas);
}
