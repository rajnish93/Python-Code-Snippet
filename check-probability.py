# Predicted Probabilities From trained model
F.softmax(model(input_ids, attention_mask), dim=1)