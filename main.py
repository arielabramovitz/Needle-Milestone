import numpy as np
import nltk as nl
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import scrapy
from scrapy.crawler import CrawlerProcess
import os


nl.download('stopwords')
nl.download('punkt')


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

def print_sorted_tfidf(dense_matrix, feature_names, filenames):
    for i, filename in enumerate(filenames):
        print(f"TF-IDF values for {filename}:")
        tfidf_scores = dense_matrix[i].tolist()[0]
        tfidf_scores = np.array(tfidf_scores)
        sorted_indices = np.argsort(tfidf_scores)[::-1]  # Sort in descending order
        for idx in sorted_indices:
            if tfidf_scores[idx] > 0:  # Print only non-zero values
                print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")
        print("\n")

if __name__ == '__main__':
    # process = CrawlerProcess()
    # process.crawl(BodyTextSpider)
    # process.start()
    stop_words = set(stopwords.words('english'))
    articles = [f'article{i}.txt' for i in range(1, 11)]
    # vectorizer = TfidfVectorizer(input='filename', stop_words='english', encoding='utf-8')
    # tfidf_matrix = vectorizer.fit_transform(articles)
    # feature_names = vectorizer.get_feature_names_out()
    # dense_tfidf_matrix = tfidf_matrix.todense()
    # print_sorted_tfidf(dense_tfidf_matrix, feature_names, articles)
    text = ""
    for article in articles:
        with open(article, "r", encoding='utf-8') as f:
            text += f"{f.read()} "

    vectorizer = TfidfVectorizer(stop_words='english', encoding='utf-8')
    tfidf_matrix = vectorizer.fit_transform([text])
    feature_names = vectorizer.get_feature_names_out()
    dense_tfidf_matrix = tfidf_matrix.todense()
    print_sorted_tfidf(dense_tfidf_matrix, feature_names, text)