# k-Nearest Neighbor (kNN) Collaborative Filtering

This project implements the k-Nearest Neighbor (kNN) algorithm to predict whether a student will answer a question correctly based on similar response patterns from other students or similar questions. It explores both **user-based** and **item-based** collaborative filtering approaches.

## Features

1. **User-Based Collaborative Filtering:**
   - Predict student performance by comparing students with similar response patterns.
   - Implemented in the `knn_impute_by_user()` function.

2. **Item-Based Collaborative Filtering:**
   - Predict student performance by comparing similar questions.
   - Implemented in the `knn_impute_by_item()` function.

3. **Parameter Tuning:**
   - Experiment with different values of `k` (e.g., 1, 6, 11, 16, 21, 26) to find the optimal number of nearest neighbors for accurate predictions.
   - The accuracy of predictions is evaluated using validation data, and the best `k` is selected dynamically based on accuracy.

4. **Comparison of Approaches:**
   - A comparison of user-based and item-based collaborative filtering is performed to determine which approach works better for this dataset.
   - The results are visualized in plots and reported in terms of accuracy.

## How It Works

### User-Based Collaborative Filtering
In user-based filtering, students who answered questions similarly are likely to perform similarly on future questions.

- **Function:** `knn_impute_by_user(matrix, valid_data, k)`
  - Uses KNN to find the `k` nearest students for each student with missing values.
  - Imputes missing values based on the neighbors' responses.
  - Evaluates the accuracy of the imputed values using validation data.

### Item-Based Collaborative Filtering
In item-based filtering, similar questions are used to predict how a student will perform on a particular question.

- **Function:** `knn_impute_by_item(matrix, valid_data, k)`
  - Transposes the matrix to treat questions as the main entity.
  - Uses KNN to find similar questions and predict missing values.
  - Transposes the matrix back to evaluate the accuracy of the predictions.

## Results

### User-Based Filtering:
- **Optimal k:** 11
- **Test Accuracy:** 0.6842

### Item-Based Filtering:
- **Chosen k:** 21
- **Test Accuracy:** 0.6816

### Comparison:
- **User-based filtering** slightly outperformed **item-based filtering**, with an accuracy of 0.6842 compared to 0.6816.
- The slight advantage of user-based filtering could be due to stronger response pattern similarities between students compared to questions.

## How to Run

1. Clone this repository.
2. Install the required dependencies (e.g., `scikit-learn`, `matplotlib`).
   ```bash
   pip install -r requirements.txt
   ```
3. Load the dataset in the `data` folder (ensure that the training, validation, and test datasets are properly formatted).
4. Run the `main()` function in the script.
   ```bash
   python knn.py
   ```

5. Observe the results and visualizations showing accuracy for different values of `k`.

## Files

- `knn.py`: The main implementation of k-Nearest Neighbor collaborative filtering (both user-based and item-based).
- `utils.py`: Utility functions for loading datasets and evaluating the accuracy of predictions.
- `data/`: Folder containing the dataset files (training, validation, and test data).

## Conclusion

While both user-based and item-based filtering performed similarly, user-based filtering showed a slight edge in this dataset. This might be due to stronger similarities in student behavior and learning patterns, making their responses easier to predict based on similar peers.