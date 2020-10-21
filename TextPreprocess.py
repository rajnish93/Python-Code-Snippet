# Remove Stopwords function
def remove_stopwords(string):
    word_list = [word.lower() for word in string.split()]
    stopwords_list = list(stopwords.words("english"))
    for word in word_list:
        if word in stopwords_list:
            word_list.remove(word)
    return ' '.join(word_list)
# Preprocess Function
def preprocess_fn(df):
    for column in ['body', 'title']:
        df[column] = df[column].map(lambda x: re.sub('\\n', ' ', str(x)))
        df[column] = df[column].map(lambda x: re.sub(r'\W', ' ', str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'https\s+|www.\s+', r'', str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'http\s+|www.\s+', r'', str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'\s+[a-zA-Z]\s+', ' ', str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'\^[a-zA-Z]\s+', ' ', str(x)))
        df[column] = df[column].map(lambda x: re.sub(r'\s+', ' ', str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\’", "\'", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"won\'t", "will not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"can\'t", "can not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"don\'t", "do not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"dont", "do not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"n\’t", " not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"n\'t", " not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"\'re", " are", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\'s", " is", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"\’d", " would", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"\d", " would", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"\'ll", " will", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\'t", " not", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"\'ve", " have", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\'m", " am", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\n", "", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\r", "", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r"[0-9]", "digit", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\'", "", str(x)))
        df[column] = df[column].map(lambda x: re.sub(r"\"", "", str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'[?|!|\'|"|#]', r'', str(x)))
        df[column] = df[column].map(
            lambda x: re.sub(r'[.|,|)|(|\|/]', r' ', str(x)))
        df[column] = df[column].apply(lambda x: remove_stopwords(x))
    return df