#pragma once
#include <cstdint>

/* MNIST DATASET HELPER CLASS HEADER
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
This class is a set of handy static functions to deal with the MNIST dataset.
This is stored in files in the data/ folder of the project and contains a big
set of training data for handwritten digits.

This class is meant for easy manipulation of the data in such set. First,
loadDataSet() should be called to load the files, and then you can easily 
obtain a pointer to the images and labels, and training values for subsets.
-------------------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------------------
*/

// Expected rows and columns to read from the dataset

#define IMG_ROWS 28u
#define IMG_COLS 28u
#define IMG_PIXELS (IMG_ROWS * IMG_COLS)

// Simple static class to handle the loading of the MNIST dataset for handwritten digits.
class MNIST
{
private:

	static inline uint8_t** trainingSetImages = nullptr;	// MNIST pointer to the training images
	static inline uint8_t** testingSetImages = nullptr;		// MNIST pointer to the testing images

	static inline uint8_t* trainingSetLabels = nullptr;		// MNIST pointer to the training labels
	static inline uint8_t* testingSetLabels = nullptr;		// MNIST pointer to the testing labels

	static inline size_t n_training_images = 0;				// Total images loaded from the training set
	static inline size_t n_testing_images = 0;				// Total images loaded from the testing set

	static MNIST helper;	// Helper to call the destructor at the end of the program

	// When the program ends the helper will call the destructor,
	// freeing the dataset if this was loaded.
	~MNIST();
	MNIST() = default;

public:

	// Loads the files for the dataset and stores the images in RAM.
	static void loadDataSet();

	// Returns the pointer to the training images as a uint8_t**.
	static uint8_t** getTrainingSetImages();

	// Returns the pointer to the training lables as an uint8_t*.
	static uint8_t* getTrainingSetLabels();

	// Returns the pointer to the testing images as a uint8_t**.
	static uint8_t** getTestingSetImages();

	// Returns the pointer to the testing lables as an uint8_t*.
	static uint8_t* getTestingSetLabels();

	// Returns the numpber of images stored in the training dataset.
	static size_t get_n_training_images();

	// Returns the numpber of images stored in the testing dataset.
	static size_t get_n_testing_images();

#ifdef _CONSOLE
	// Simple function to print the images in the console.
	static void consolePrint(uint8_t* image);
#endif
};


#define IMAGE_DIM 784

enum Set
{
	TESTING,
	TRAINING
};

class NumberRecognition
{
private:

	static inline unsigned* trainingLabels;
	static inline unsigned* testingLabels;

	static inline float** trainingImages;
	static inline float** testingImages;

	static inline float** trainingValues;
	static inline float** testingValues;

	static inline unsigned n_training;
	static inline unsigned n_testing;

	static NumberRecognition loader;

	NumberRecognition();
	~NumberRecognition();
public:

	static float** getImages(Set test_train, size_t start_idx, size_t end_idx);

	static unsigned* getLabels(Set test_train, size_t start_idx, size_t end_idx);

	static void printImage(Set test_train, size_t idx);

	static unsigned getSize(Set test_train);
};