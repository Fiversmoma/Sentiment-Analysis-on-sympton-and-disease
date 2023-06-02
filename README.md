# Sentiment Analysis using Naive Bayes, Conv1D, and Transfer Learning (USE)

This project focuses on sentiment analysis using three different models: Naive Bayes, Conv1D, and Transfer Learning with Universal Sentence Encoder (USE). The goal is to classify text data into positive or negative sentiment.
Table of Contents

    Project Description
    Installation
    Usage
    Models
    Results
    Contributing
    License

Project Description

Sentiment analysis is a text classification task that aims to determine the sentiment expressed in a given text, whether it is positive, negative, or neutral. In this project, we explored three different models to perform sentiment analysis: Naive Bayes, Conv1D (1D Convolutional Neural Network), and Transfer Learning with Universal Sentence Encoder (USE).

The project involved data preprocessing, model training, evaluation, and comparison of the three models. The Naive Bayes model is a probabilistic classifier, Conv1D is a deep learning model for sequence classification, and Transfer Learning with USE leverages pre-trained language embeddings for sentiment analysis.
Installation

To run this project, please follow these steps:

    Clone the repository to your local machine.
    Install the required dependencies by running pip install -r requirements.txt.
    Prepare your dataset and ensure it is in a suitable format for training the models.

Usage

To use this project, follow these steps:

    Ensure you have the necessary dataset or text data for sentiment analysis.
    Modify the code to load and preprocess your specific dataset.
    Run the models individually or compare their performance by executing the appropriate code snippets.
    Analyze the results and evaluate the accuracy, loss, or any other relevant metrics.

Note: This project assumes you have basic knowledge of machine learning concepts and familiarity with Python and the required libraries.
Models

The project utilizes the following models for sentiment analysis:

    Naive Bayes:
        A probabilistic classifier based on Bayes' theorem.
        It assumes that features are conditionally independent.
        Fast training and prediction times.
        Relatively simple implementation and interpretability.

    Conv1D:
        A 1D Convolutional Neural Network model for sequence classification.
        Learns local patterns and dependencies within the input sequences.
        Suitable for text data and capturing spatial patterns.
        More complex model architecture and longer training times compared to Naive Bayes.

    Transfer Learning with Universal Sentence Encoder (USE):
        Utilizes the Universal Sentence Encoder for sentence-level embeddings.
        Leverages pre-trained language models to transfer knowledge to the sentiment analysis task.
        Offers high-level, contextualized representations of text data.
        Requires access to pre-trained USE models and additional computational resources.

Results

The evaluation results for the three models are as follows:

    Naive Bayes:
        Accuracy: 93.33%

    Conv1D:
        Accuracy: 98.33%
        Loss: 0.0925

    Transfer Learning (USE):
        Accuracy: 97.50%
        Loss: 0.0679

Based on the results, Conv1D achieved the highest accuracy, outperforming Naive Bayes and Transfer Learning with USE. However, the choice of the most suitable model depends on various factors, such as model complexity, computational requirements, and specific project requirements.
Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please feel free to submit a pull request or open an issue on the GitHub repository.
