#include "MNIST.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// Paths for the MNIST files

#define MNIST_TRAINING_IMAGES_PATH MNIST_PATH "train-images.idx3-ubyte"
#define MNIST_TRAINING_LABELS_PATH MNIST_PATH "train-labels.idx1-ubyte"

#define MNIST_TESTING_IMAGES_PATH MNIST_PATH "t10k-images.idx3-ubyte"
#define MNIST_TESTING_LABELS_PATH MNIST_PATH "t10k-labels.idx1-ubyte"

// Static variables definition

MNIST MNIST::helper;

// When the program ends the helper will call the destructor,
// freeing the dataset if this was loaded.

MNIST::~MNIST()
{
    if (trainingSetImages)
    {
        for (unsigned int i = 0; i < n_training_images; ++i)
            if (trainingSetImages[i])
                free(trainingSetImages[i]);

        free(trainingSetImages);
        trainingSetImages = nullptr;
    }

    if (testingSetImages)
    {
        for (unsigned int i = 0; i < n_testing_images; ++i)
            if (testingSetImages[i])
                free(testingSetImages[i]);

        free(testingSetImages);
        testingSetImages = nullptr;
    }

    if (trainingSetLabels)
    {
        free(trainingSetLabels);
        trainingSetLabels = nullptr;
    }

    if (testingSetLabels)
    {
        free(testingSetLabels);
        testingSetLabels = nullptr;
    }
}

// Static helper to read 32bits from a file.

static inline uint32_t read_be_u32(FILE* f)
{
    uint8_t b[4];
    if (fread(b, 1, 4, f) != 4)
        return 0xFFFFFFFFu;

    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | ((uint32_t)b[3]);
}

// Loads the files for the dataset and stores the images in RAM.

void MNIST::loadDataSet()
{
    // If already loaded, do nothing.
    if (trainingSetImages)
        return;

    // Open files
    FILE* training_images_file = fopen(MNIST_TRAINING_IMAGES_PATH, "rb");
    if (!training_images_file)
    {
        fprintf(stderr, "[MNIST] Failed to open '%s'\n", MNIST_TRAINING_IMAGES_PATH);
        return;
    }

    FILE* testing_images_file = fopen(MNIST_TESTING_IMAGES_PATH, "rb");
    if (!testing_images_file)
    {
        fprintf(stderr, "[MNIST] Failed to open '%s'\n", MNIST_TESTING_IMAGES_PATH);
        fclose(training_images_file);
        return;
    }

    FILE* training_labels_file = fopen(MNIST_TRAINING_LABELS_PATH, "rb");
    if (!training_labels_file)
    {
        fprintf(stderr, "[MNIST] Failed to open '%s'\n", MNIST_TRAINING_LABELS_PATH);
        fclose(training_images_file);
        fclose(testing_images_file);
        return;
    }

    FILE* testing_labels_file = fopen(MNIST_TESTING_LABELS_PATH, "rb");
    if (!testing_labels_file)
    {
        fprintf(stderr, "[MNIST] Failed to open '%s'\n", MNIST_TESTING_LABELS_PATH);
        fclose(training_images_file);
        fclose(testing_images_file);
        fclose(training_labels_file);
        return;
    }

    // Read Headers
    // Training
    const uint32_t magic_i0 = read_be_u32(training_images_file);
    const uint32_t count_i0 = read_be_u32(training_images_file);
    const uint32_t rows0 = read_be_u32(training_images_file);
    const uint32_t cols0 = read_be_u32(training_images_file);

    const uint32_t magic_l0 = read_be_u32(training_labels_file);
    const uint32_t count_l0 = read_be_u32(training_labels_file);
    // Testing
    const uint32_t magic_i1 = read_be_u32(testing_images_file);
    const uint32_t count_i1 = read_be_u32(testing_images_file);
    const uint32_t rows1 = read_be_u32(testing_images_file);
    const uint32_t cols1 = read_be_u32(testing_images_file);

    const uint32_t magic_l1 = read_be_u32(testing_labels_file);
    const uint32_t count_l1 = read_be_u32(testing_labels_file);

    // Make sure there is nothing weird going on here
    if (magic_i0 != 0x00000803u)
    {
        fprintf(stderr, "[MNIST] Bad image magic: 0x%08X (expected 0x00000803)\n", magic_i0);
        goto cleanup;
    }
    if (magic_i1 != 0x00000803u)
    {
        fprintf(stderr, "[MNIST] Bad image magic: 0x%08X (expected 0x00000803)\n", magic_i1);
        goto cleanup;
    }
    if (rows0 != IMG_ROWS || cols0 != IMG_COLS)
    {
        fprintf(stderr, "[MNIST] Unexpected image dims: %u x %u (expected 28 x 28)\n", rows0, cols0);
        goto cleanup;
    }
    if (rows1 != IMG_ROWS || cols1 != IMG_COLS)
    {
        fprintf(stderr, "[MNIST] Unexpected image dims: %u x %u (expected 28 x 28)\n", rows1, cols1);
        goto cleanup;
    }
    if (count_i0 == 0)
    {
        fprintf(stderr, "[MNIST] No images in file.\n");
        goto cleanup;
    }
    if (count_i1 == 0)
    {
        fprintf(stderr, "[MNIST] No images in file.\n");
        goto cleanup;
    }
    if (magic_l0 != 0x00000801u)
    {
        fprintf(stderr, "[MNIST] Bad label magic: 0x%08X (expected 0x00000801)\n", magic_l0);
        goto cleanup;
    }
    if (magic_l1 != 0x00000801u)
    {
        fprintf(stderr, "[MNIST] Bad label magic: 0x%08X (expected 0x00000801)\n", magic_l1);
        goto cleanup;
    }
    if (count_l0 != count_i0)
    {
        fprintf(stderr, "[MNIST] Count mismatch (images=%u, labels=%u)\n", count_i0, count_l0);
        goto cleanup;
    }
    if (count_l1 != count_i1)
    {
        fprintf(stderr, "[MNIST] Count mismatch (images=%u, labels=%u)\n", count_i1, count_l1);
        goto cleanup;
    }


    n_training_images = count_i0;
    n_testing_images = count_i1;

    trainingSetLabels = (uint8_t*)calloc(n_training_images, sizeof(char));
    trainingSetImages = (uint8_t**)calloc(n_training_images, sizeof(void*));

    testingSetLabels = (uint8_t*)calloc(n_testing_images, sizeof(char));
    testingSetImages = (uint8_t**)calloc(n_testing_images, sizeof(void*));

    // Read labels (all at once)
    if (fread(trainingSetLabels, 1, n_training_images, training_labels_file) != n_training_images)
    {
        fprintf(stderr, "[MNIST] Unexpected EOF while reading labels.\n");
        goto cleanup;
    }

    if (fread(testingSetLabels, 1, n_testing_images, testing_labels_file) != n_testing_images)
    {
        fprintf(stderr, "[MNIST] Unexpected EOF while reading labels.\n");
        goto cleanup;
    }

    // Read images (one by one)
    for (unsigned int i = 0; i < n_training_images; ++i)
    {
        trainingSetImages[i] = (uint8_t*)calloc(IMG_PIXELS, sizeof(uint8_t));

        if (fread(trainingSetImages[i], 1, IMG_PIXELS, training_images_file) != IMG_PIXELS)
        {
            fprintf(stderr, "[MNIST] Unexpected EOF while reading image %u.\n", i);
            goto cleanup;
        }
    }

    for (unsigned int i = 0; i < n_testing_images; ++i)
    {
        testingSetImages[i] = (uint8_t*)calloc(IMG_PIXELS, sizeof(uint8_t));

        if (fread(testingSetImages[i], 1, IMG_PIXELS, testing_images_file) != IMG_PIXELS)
        {
            fprintf(stderr, "[MNIST] Unexpected EOF while reading image %u.\n", i);
            goto cleanup;
        }
    }

    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(testing_images_file);
    fclose(testing_labels_file);
    return;

cleanup:

    if (trainingSetImages)
    {
        for (unsigned int i = 0; i < n_training_images; ++i)
            if (trainingSetImages[i])
                free(trainingSetImages[i]);

        free(trainingSetImages);
        trainingSetImages = nullptr;
    }

    if (testingSetImages)
    {
        for (unsigned int i = 0; i < n_testing_images; ++i)
            if (testingSetImages[i])
                free(testingSetImages[i]);

        free(testingSetImages);
        testingSetImages = nullptr;
    }

    if (trainingSetLabels)
    {
        free(trainingSetLabels);
        trainingSetLabels = nullptr;
    }

    if (testingSetLabels)
    {
        free(testingSetLabels);
        testingSetLabels = nullptr;
    }

    n_training_images = 0;
    n_testing_images = 0;

    fclose(training_images_file);
    fclose(training_labels_file);
    fclose(testing_images_file);
    fclose(testing_labels_file);
}

// Returns the pointer to the training images as a uint8_t**.

uint8_t** MNIST::getTrainingSetImages()
{
    return trainingSetImages;
}

// Returns the pointer to the training lables as an uint8_t*.

uint8_t* MNIST::getTrainingSetLabels()
{
    return trainingSetLabels;
}

// Returns the pointer to the testing images as a uint8_t**.

uint8_t** MNIST::getTestingSetImages()
{
    return testingSetImages;
}

// Returns the pointer to the testing lables as an uint8_t*.

uint8_t* MNIST::getTestingSetLabels()
{
    return testingSetLabels;
}

// Returns the numpber of images stored in the training dataset.

size_t MNIST::get_n_training_images()
{
    return n_training_images;
}

// Returns the numpber of images stored in the testing dataset.

size_t MNIST::get_n_testing_images()
{
    return n_testing_images;
}

#ifdef _CONSOLE
// Simple function to print the images in the console.

void MNIST::consolePrint(uint8_t* image)
{
    system("color");
    printf("\033[0;34m");
    for (unsigned int r = 0; r < IMG_ROWS; r++)
    {
        for (unsigned int c = 0; c < IMG_COLS; c++)
        {
            if (image[c + r * IMG_COLS])
                printf("\033[0;34m");
            else
                printf("\033[0;31m");

            printf("%c%c", 219, 219);
        }
        printf("\n");
    }
    printf("\033[0m");
}
#endif

/*
-------------------------------------------------------------------------------------------------------
Constructor / Destructor
-------------------------------------------------------------------------------------------------------
*/

NumberRecognition NumberRecognition::loader;

NumberRecognition::NumberRecognition()
{
    MNIST::loadDataSet();
    if (!MNIST::getTrainingSetImages())
        abort();

    n_training = (unsigned)MNIST::get_n_training_images();
    n_testing = (unsigned)MNIST::get_n_testing_images();

    trainingImages = (float**)calloc(n_training, sizeof(float*));
    trainingValues = (float**)calloc(n_training, sizeof(float*));
    trainingLabels = (unsigned*)calloc(n_training, sizeof(unsigned));

    testingImages = (float**)calloc(n_testing, sizeof(float*));
    testingValues = (float**)calloc(n_testing, sizeof(float*));
    testingLabels = (unsigned*)calloc(n_testing, sizeof(unsigned));

    uint8_t* training_labels = MNIST::getTrainingSetLabels();
    uint8_t* testing_labels = MNIST::getTestingSetLabels();

    for (unsigned i = 0; i < n_training; i++)
        trainingLabels[i] = (unsigned)training_labels[i];

    for (unsigned i = 0; i < n_testing; i++)
        testingLabels[i] = (unsigned)testing_labels[i];
}

NumberRecognition::~NumberRecognition()
{

    free(trainingLabels);
    free(testingLabels);

    for (unsigned i = 0; i < n_training; i++)
    {
        if (trainingValues[i])
            free(trainingValues[i]);

        if (trainingImages[i])
            free(trainingImages[i]);
    }

    for (unsigned i = 0; i < n_testing; i++)
    {
        if (testingValues[i])
            free(testingValues[i]);

        if (testingImages[i])
            free(testingImages[i]);
    }


    free(trainingImages);
    free(trainingValues);

    free(testingImages);
    free(testingValues);

}

/*
-------------------------------------------------------------------------------------------------------
User end functions
-------------------------------------------------------------------------------------------------------
*/

float** NumberRecognition::getImages(Set test_train, size_t start_idx, size_t end_idx)
{
    uint8_t** raw_images;
    float** my_images;

    switch (test_train)
    {
    case TESTING:
        raw_images = MNIST::getTestingSetImages();
        my_images = testingImages;
        break;

    case TRAINING:
        raw_images = MNIST::getTrainingSetImages();
        my_images = trainingImages;
        break;

    default:
        return nullptr;
    }

    size_t n_data = end_idx - start_idx;

    for (size_t i = 0; i < n_data; i++)
    {
        size_t idx = i + start_idx;
        uint8_t* image = raw_images[idx];

        if (my_images[idx])
            continue;

        my_images[idx] = (float*)calloc(IMAGE_DIM, sizeof(float));

        for (unsigned i = 0; i < IMAGE_DIM; i++)
            my_images[idx][i] = float(image[i]) / 256.f;
    }

    return &my_images[start_idx];
}

unsigned* NumberRecognition::getLabels(Set test_train, size_t start_idx, size_t end_idx)
{
    switch (test_train)
    {
    case TESTING:
        return testingLabels;

    case TRAINING:
        return trainingLabels;

    default:
        return nullptr;
    }
}

void NumberRecognition::printImage(Set test_train, size_t idx)
{
    switch (test_train)
    {
    case TESTING:
        MNIST::consolePrint(MNIST::getTestingSetImages()[idx]);
        return;

    case TRAINING:
        MNIST::consolePrint(MNIST::getTrainingSetImages()[idx]);
        return;

    default:
        return;
    }
}

unsigned NumberRecognition::getSize(Set test_train)
{
    switch (test_train)
    {
    case TESTING:
        return n_testing;

    case TRAINING:
        return n_training;

    default:
        return 0U;
    }
}
