import os
import librosa
import numpy as np
import cv2
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, train_test_split
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
from skimage import io
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
# Function stubs for feature extraction and combination
def extract_voice_features(directory):
    features = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):  # Assuming audio files are in .wav format
            filepath = os.path.join(directory, filename)
            try:
                # Load audio file
                y, sr = librosa.load(filepath, sr=None)
                # Extract features (e.g., MFCCs)
                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                # Calculate mean of each MFCC coefficient
                mean_mfccs = np.mean(mfccs, axis=1)
                # Append features and label
                features.append(mean_mfccs)
                labels.append(directory.split('/')[-1])   #Assuming directory structure: males/ or females/
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    return np.array(features)


def extract_face_features(directory):
    X= []
    #y = []

    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory does not exist: {directory}")
        return np.array(X), y

    # Determine if the directory contains subdirectories or directly images
    contains_direct_images = any(file.endswith(".jpg") for file in os.listdir(directory))
    if contains_direct_images:
        # Process files directly in the main directory
        files_to_process = os.listdir(directory)
        directory_label = os.path.basename(directory)
        process_files(files_to_process, directory,X)
    '''else:
        # Process files in subdirectories
        for subfolder in os.listdir(landmark_directory):
            subfolder_path = os.path.join(landmark_directory, subfolder)
            if os.path.isdir(subfolder_path):
                files_to_process = os.listdir(subfolder_path)
                process_files(files_to_process, subfolder_path, subfolder, X, y)
            else:
                print(f"Expected directory, found file: {subfolder_path}")'''

    return np.array(X)

def process_files(files, path, X,):
    for file in files:
        if file.endswith(".jpg"):
            image_path = os.path.join(path, file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Ensure grayscale
            if image is not None:
                try:
                    if len(image.shape) != 2:  # Check if the image is truly 2D
                        print(f"Image not 2D (has shape {image.shape}): {image_path}")
                        continue

                    image = cv2.resize(image, (64, 128))  # Resize image
                    fd = hog(image, orientations=8, pixels_per_cell=(16, 16),
                             cells_per_block=(1, 1), visualize=False)
                    X.append(fd)
                    #y.append(label)
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
            else:
                print(f"Failed to load image: {image_path}")
        else:
            print(f"Skipped non-JPG file: {file}")




def standardize_features(features, target_length):
    """Standardize the length of feature vectors and ensure they are numpy arrays."""
    standardized_features = []
    for feature in features:
        # Ensure feature is a numpy array
        feature = np.array(feature).flatten()
        if len(feature) > target_length:
            feature = feature[:target_length]
        elif len(feature) < target_length:
            padding = np.zeros(target_length - len(feature))
            feature = np.concatenate([feature, padding])

        standardized_features.append(feature)
    return np.array(standardized_features)


def combine_features(voice_features, face_features):
    """Combine standardized voice and face features into a single feature set."""
    if len(voice_features) != len(face_features):
        raise ValueError("Mismatch in the number of voice and face features")

    combined_features = []
    for voice, face in zip(voice_features, face_features):
        combined_feature = np.concatenate([voice, face])
        combined_features.append(combined_feature)

    return np.vstack(combined_features)



print("All landmarks loaded")
voice_features_male = extract_voice_features("multiVoiceM")
voice_features_female = extract_voice_features("multiVoiceF")
face_features_male = extract_face_features("multiFaceM")
face_features_female = extract_face_features("multiFaceF")
# Standardize features
standardized_voice_features_male = standardize_features(voice_features_male, target_length=650)
standardized_face_features_male = standardize_features(face_features_male, target_length=12800)
standardized_voice_features_female = standardize_features(voice_features_female, target_length=650)
standardized_face_features_female = standardize_features(face_features_female, target_length=12800)
# Combine features
features_male = combine_features(standardized_voice_features_male, standardized_face_features_male)
features_female = combine_features(standardized_voice_features_female, standardized_face_features_female)

# Convert features to float and ensure numpy array
features_male = np.array(features_male).astype(float)
features_female = np.array(features_female).astype(float)

# Combine male and female features and labels
labels_female = [1] * len(features_female)
labels_male = [0] * len(features_male)

features = np.concatenate((features_male, features_female), axis=0)
labels = np.array([0] * len(features_male) + [1] * len(features_female))
'''
small_features, _, small_labels, _ = train_test_split(
    features, labels, test_size=0.9, stratify=labels, random_state=42)
gbc = GradientBoostingClassifier(random_state=42)
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(gbc, param_grid, cv=5, verbose=3, n_jobs=-1, error_score='raise')

try:
    grid_search.fit(small_features, small_labels)
except ValueError as e:
    print(f"An error occurred during grid search: {e}")

best_model = grid_search.best_estimator_
# Evaluate model with cross-validation
cv_scores = cross_val_score(best_model, features, labels, cv=5)
print(f'Cross-validated scores: {cv_scores}')
print(f'Average CV score: {np.mean(cv_scores) * 100:.2f}%')
'''


model = RandomForestClassifier(n_estimators=50, min_samples_leaf=5, random_state=42).fit(features, labels)
X_train_male, X_test_male, y_train_male, y_test_male = train_test_split(features_male, labels_male, test_size=0.2, random_state=42)
X_train_female, X_test_female, y_train_female, y_test_female = train_test_split(features_female, labels_female, test_size=0.2, random_state=42)

# Fit the model on combined training data and evaluate on separate test sets
model.fit(np.concatenate((X_train_male, X_train_female)), np.concatenate((y_train_male, y_train_female)))

male_accuracy = accuracy_score(y_test_male, model.predict(X_test_male))
female_accuracy = accuracy_score(y_test_female, model.predict(X_test_female))

print(f'Male Specific Accuracy (proper test split): {male_accuracy * 100:.2f}%')
print(f'Female Specific Accuracy (proper test split): {female_accuracy * 100:.2f}%')

'''

'''
# Combine the train and test sets for both genders
X_train = np.concatenate((X_train_male, X_train_female))
y_train = np.concatenate((y_train_male, y_train_female))
X_test = np.concatenate((X_test_male, X_test_female))
y_test = np.concatenate((y_test_male, y_test_female))

# Fit the model on the combined training data
model.fit(X_train, y_train)

# Predict on the combined test set
y_pred = model.predict(X_test)

# Calculate the overall accuracy
overall_accuracy = accuracy_score(y_test, y_pred)
print(f'Overall Accuracy: {overall_accuracy * 100:.2f}%')



# Combine and transform data using PCA
pca = PCA(n_components=2)  # reduce to two dimensions for visualization

projected = pca.fit_transform(features.astype(float))
labels_combined = np.array([0] * len(features_male) + [1] * len(features_female))

# Plotting the results
plt.figure(figsize=(8, 6))
scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels_combined, alpha=0.5, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Gender (0=Male, 1=Female)')
plt.title('PCA of Features by Gender')
plt.grid(True)
plt.show()



mean_male = np.mean(features_male, axis=0)
std_dev_male = np.std(features_male, axis=0)

# Calculate means and standard deviations for female features
mean_female = np.mean(features_female, axis=0)
std_dev_female = np.std(features_female, axis=0)

# Printing the results
print("Male Features Mean:\n", mean_male)
print("Male Features Standard Deviation:\n", std_dev_male)

print("Female Features Mean:\n", mean_female)
print("Female Features Standard Deviation:\n", std_dev_female)

# Calculate differences in means and standard deviations
diff_means = np.abs(mean_male - mean_female)
diff_stds = np.abs(std_dev_male - std_dev_female)

# Plotting differences in means
plt.figure(figsize=(10, 4))
plt.bar(range(len(diff_means)), diff_means)
plt.title('Difference in Means Between Genders in a Multimodal System ')
plt.xlabel('Feature Index')
plt.ylabel('Absolute Difference in Mean')
plt.grid(True)
plt.show()

# Plotting differences in standard deviations
plt.figure(figsize=(10, 4))
plt.bar(range(len(diff_stds)), diff_stds)
plt.title('Difference in Standard Deviations Between Genders: Multimodal System')
plt.xlabel('Feature Index')
plt.ylabel('Absolute Difference in Standard Deviation')
plt.grid(True)
plt.show()