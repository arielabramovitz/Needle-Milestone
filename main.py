import numpy as np
import nltk as nl
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import scrapy
from scrapy.crawler import CrawlerProcess
import os
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.stem import porter
from wordcloud import WordCloud

nl.download('stopwords')
nl.download('punkt')
nl.download('omw-1.4')
nl.download('wordnet')


class BodyTextSpider(scrapy.Spider):
    name = "bodytext"
    urls = [

    ]

    def start_requests(self):
        with open('urls.txt', 'r') as f:
            urls = f.readlines()
        for url in urls:
            url = url.strip()
            if url:
                yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response, **kwargs):
        # Extract body text
        body_text = ''.join(response.xpath('//body//text()').getall()).strip()

        # Ensure results directory exists
        if not os.path.exists('results'):
            os.makedirs('results')

        # Save the result to a file named after the URL
        url = response.url.replace('http://', '').replace('https://', '').replace('/', '_')
        with open(f'results/{url}.txt', 'w', encoding='utf-8') as f:
            f.write(body_text)

        self.log(f'Saved file {url}.txt')


def preprocess(text):
    tokens = word_tokenize(text)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words and word.isalnum()]
    return ' '.join(filtered_tokens)


# def print_sorted_tfidf(dense_matrix, feature_names, filenames):
#     for i, filename in enumerate(filenames):
#         print(f"TF-IDF values for {filename}:")
#         tfidf_scores = dense_matrix[i].tolist()[0]
#         tfidf_scores = np.array(tfidf_scores)
#         sorted_indices = np.argsort(tfidf_scores)[::-1]  # Sort in descending order
#         for idx in sorted_indices:
#             if tfidf_scores[idx] > 0:  # Print only non-zero values
#                 print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")
#         print("\n")


def get_tfidf_score_and_indices(dense_matrix):
    tfidf_scores = dense_matrix[0].tolist()[0]
    tfidf_scores = np.array(tfidf_scores)
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    return tfidf_scores, sorted_indices


def print_sorted_tfidf(dense_matrix, feature_names, filename):
    print(f"TF-IDF values for {filename}:")
    tfidf_scores, sorted_indices = get_tfidf_score_and_indices(dense_matrix)
    # tfidf_scores = dense_matrix[0].tolist()[0]
    # tfidf_scores = np.array(tfidf_scores)
    # sorted_indices = np.argsort(tfidf_scores)[::-1]  # Sort in descending order
    for idx in sorted_indices:
        if tfidf_scores[idx] > 0:  # Print only non-zero values
            print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")
    print("\n")


def plot_reality_check(dense_matrix, feature_names, top_n_words):
    tfidf_scores, sorted_indices = get_tfidf_score_and_indices(dense_matrix)
    sorted_indices = sorted_indices[:top_n_words]
    words = []
    scores = []
    for idx in sorted_indices:
        words.append(feature_names[idx])
        scores.append(tfidf_scores[idx])
    words = np.array(words)
    scores = np.array(scores)
    plt.figure(figsize=(12, 6))
    plt.bar(range(top_n_words), scores, align='center')
    plt.xticks(range(top_n_words), words, rotation=45, ha='right')
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.title(f'Top {top_n_words} words by TF-IDF score')
    plt.tight_layout()
    plt.show()


def get_articles_list(folders):
    articles_lst = []
    for folder in folders:
        jerusalem_string = "jerusalem_article"
        times_string = "times_article"
        range_articles = 11
        articles_lst += [
            f'{folder}\{name}{i}.txt'
            for name in [jerusalem_string, times_string]
            for i in range(1, range_articles)
        ]
    return articles_lst


if __name__ == '__main__':
    # process = CrawlerProcess()
    # process.crawl(BodyTextSpider)
    # process.start()

    stop_words = set(stopwords.words('english'))
    articles_folders = ["elections2019b", "elections2020",
                        "elections2021", "elections2022"]
    articles = get_articles_list(articles_folders)

    # vectorizer = TfidfVectorizer(input='filename', stop_words='english', encoding='utf-8')
    # tfidf_matrix = vectorizer.fit_transform(articles)
    # feature_names = vectorizer.get_feature_names_out()
    # dense_tfidf_matrix = tfidf_matrix.todense()
    # print_sorted_tfidf(dense_tfidf_matrix, feature_names, articles)

    stemmer = porter.PorterStemmer()
    tok = RegexpTokenizer(r"\b\w+(?:[`'’]\w+)?(?!'s)\b")
    text = ""

    exclusions = ["tikva hadasha", "yesh atid", "new hope"]
    total_len = 0
    for article in articles:
        with open(article, "r", encoding='utf-8') as f:
            a = f.read().lower()

            for exc in exclusions:
                a = a.replace(exc, "_".join(exc.split(" ")))
            tokens = tok.tokenize(a)
            for token in tokens:
                if token.isnumeric() or "." in token:
                    if int(token) < 1000 or int(token) > 2100:
                        tokens.remove(token)
            # tokens = [stemmer.stem(word) for word in tokens]
            text += " " + " ".join(tokens)
    text = text.replace("'s", "")
    text = text.replace("’s", "")
    vectorizer = TfidfVectorizer(token_pattern=r"\b\w+(?:[`'’]\w+)?(?!'s)\b",
                                 stop_words=list(stop_words), encoding='utf-8')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    new_features = []
    for feature in feature_names:
        if "_" in feature:
            feature = feature.replace("_", " ")
        new_features.append(feature)
    dense_tfidf_matrix = tfidf_matrix.todense()
    # print_sorted_tfidf(dense_tfidf_matrix, new_features, "All text")
    # print_sorted_tfidf(dense_tfidf_matrix, feature_names, text)
    plot_reality_check(dense_tfidf_matrix, new_features, 30)
    tf_idf_dict = {new_features[i]: dense_tfidf_matrix[0, i] for i in range(
        len(new_features))}
    wordcloud = WordCloud(width=800, height=400, background_color='white'
                          ).generate_from_frequencies(tf_idf_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
