# Deep Learning Project for Land Use/Land Cover Change and Wildfire Prediction

## Setup and Environment
- [x] Create project directory structure
- [x] Install required Python libraries (PyTorch, Transformers, OpenCV)
- [x] Test GPU availability (Not available - will use CPU)

## Data Collection and Preparation
- [x] Download UC Merced dataset
- [x] Download ESA World Cover dataset (partial - N30W060 tile)
- [x] Access Landsat 8-9 images (NBR image from NASA Disasters Program)
- [ ] Preprocess satellite imagery
- [ ] Create training/validation/test splits

## Model Architecture Research
- [x] Research U-Net architecture
- [x] Explore Segment Anything Model (SAM) integration
- [x] Explore CLIP model integration
- [x] Decide on final foundation model approach

## Model Implementation
- [x] Implement U-Net architecture
- [x] Integrate with chosen foundation model (SAM2)
- [x] Implement data loading and augmentation pipeline
- [x] Set up training configuration

## Training and Validation
- [x] Train model on prepared datasets
- [x] Monitor training metrics
- [x] Perform validation
- [x] Fine-tune hyperparameters

## Evaluation
- [x] Evaluate model performance on test set
- [x] Generate performance metrics
- [x] Create visualizations of predictions

## Prediction System
- [x] Develop inference pipeline
- [x] Create land use/land cover change prediction system
- [x] Create wildfire prediction system
- [x] Test system with new data

## Documentation and Delivery
- [x] Document model architecture
- [x] Document training process and results
- [x] Create user guide for prediction system
- [x] Prepare final report and presentation
