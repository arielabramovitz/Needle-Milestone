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

health_words = [
    'health', 'medicine', 'hospital', 'doctor', 'nurse', 'medical',
    'disease', 'treatment', 'vaccine', 'mental', 'wellness', 'clinic',
    'surgery', 'diagnosis', 'nutrition', 'healthcare', 'illness',
    'epidemic', 'pandemic', 'therapy', 'fitness', 'exercise', 'hygiene',
    'pharmacy', 'patient', 'immunity', 'infection', 'symptom', 'prevention']

crime_words = [
    'crime', 'criminal', 'theft', 'murder', 'assault', 'robbery',
    'fraud', 'violence', 'homicide', 'burglary', 'arrest', 'police',
    'law', 'justice', 'illegal', 'offense', 'penalty', 'punishment',
    'court', 'trial', 'conviction', 'prison', 'jail', 'felony',
    'misdemeanor', 'vandalism', 'smuggling', 'drug', 'gang', 'cybercrime']

education_words = [
    'education', 'school', 'university', 'college', 'student', 'teacher',
    'academic', 'curriculum', 'lecture', 'exam', 'degree', 'diploma',
    'learning', 'scholarship', 'tuition', 'classroom', 'literacy',
    'enrollment', 'graduate', 'study', 'syllabus', 'assignment',
    'research', 'institution', 'faculty', 'course', 'seminar', 'training',
    'evaluation', 'knowledge']

defense_words = [
    'defense', 'military', 'army', 'navy', 'air force', 'security',
    'war', 'weapon', 'conflict', 'battle', 'soldier', 'troop',
    'strategy', 'tactic', 'border', 'protection', 'national security',
    'alliance', 'missile', 'nuclear', 'defense minister', 'army chief',
    'intelligence', 'surveillance', 'combat', 'operation', 'patrol',
    'deployment', 'fortification', 'peacekeeping']

foreign_relations_words = [
    'diplomacy', 'foreign', 'international', 'relations', 'ambassador',
    'embassy', 'treaty', 'agreement', 'ally', 'negotiation', 'trade',
    'export', 'import', 'sanction', 'policy', 'summit', 'cooperation',
    'conflict', 'global', 'united nations', 'european union', 'nato',
    'partnership', 'dialogue', 'foreign minister', 'geopolitics',
    'bilateral', 'multilateral', 'consulate', 'visa']

religion_words = [
    'religion', 'faith', 'church', 'mosque', 'synagogue', 'temple',
    'god', 'spiritual', 'belief', 'worship', 'prayer', 'bible', 'quran',
    'torah', 'clergy', 'pastor', 'imam', 'rabbi', 'ritual', 'ceremony',
    'doctrine', 'theology', 'pilgrimage', 'saint', 'divine', 'monk',
    'nun', 'religious', 'sect', 'denomination']

candidate_names_2022 = {
    "Likud": "Benjamin Netanyahu",
    "Yesh Atid": "Yair Lapid",
    "Shas": "Aryeh Deri",
    "Blue and White": "Benny Gantz",
    "Yamina": "Ayelet Shaked",

}

candidate_names_2021 = {
    "Likud": "Benjamin Netanyahu",
    "Yesh Atid–Telem": "Yair Lapid, Moshe Ya'alon",
    "Blue and White": "Benny Gantz",
    "Derekh Eretz": "Yoaz Hendel, Zvi Hauser",
    "Joint List": "Ayman Odeh",

}

candidate_names_2020 = {
    "Blue and White": "Benny Gantz",
    "Likud": "Benjamin Netanyahu",
    "Joint List": "Ayman Odeh",
    "Shas": "Aryeh Deri",
    "Yisrael Beiteinu": "Avigdor Lieberman",

}

candidate_names_2019a = {
    "Likud": "Benjamin Netanyahu",
    "Labor": "Avi Gabbay",
    "Hatnua": "Tzipi Livni",
    "Joint List": "Ayman Odeh",
    "Ta'al": "Ahmad Tibi",

}

candidate_names_2019b = {
    "Likud": "Benjamin Netanyahu",
    "Blue and White": "Benny Gantz, Yair Lapid",
    "Shas": "Aryeh Deri",
    "United Torah Judaism": "Yaakov Litzman",
    "Hadash–Ta'al": "Ayman Odeh",

}

candidate_names_2022_keys = [
    "Likud",
    "Yesh Atid",
    "Shas",
    "Blue and White",
    "Yamina",

]

candidate_names_2021_keys = [
    "Likud",
    "Yesh Atid",
    "Blue and White",
    "Derekh Eretz",
    "Joint List",

]

candidate_names_2020_keys = [
    "Blue and White",
    "Likud",
    "Joint List",
    "Shas",
    "Yisrael Beiteinu",

]

candidate_names_2019a_keys = [
    "Likud",
    "Labor",
    "Hatnua",
    "Joint List",
    "Ta'al",
]

candidate_names_2019b_keys = [
    "Likud",
    "Blue and White",
    "Shas",
    "United Torah Judaism",
    "Hadash–Ta'al",
]


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


def get_articles_list(folders, news_lst):
    articles_lst = []
    for folder in folders:
        range_articles = 11
        articles_lst += [
            f'{folder}\{name}{i}.txt'
            for name in news_lst
            for i in range(1, range_articles)
        ]
    return articles_lst


def get_election_list(folder, news_lst):
    election_lst = []
    range_articles = 11
    election_lst += [
        f'{folder}\{name}{i}.txt'
        for name in news_lst
        for i in range(1, range_articles)
    ]
    return election_lst


def get_all_elections_list(folders, news_lst):
    elections_lst = []
    for folder in folders:
        elections_lst.append(get_election_list(folder, news_lst))
    elect_2019a = elections_lst[0]
    elect_2019b = elections_lst[1]
    elect_2020 = elections_lst[2]
    elect_2021 = elections_lst[3]
    elect_2022 = elections_lst[4]
    return elect_2019a, elect_2019b, elect_2020, elect_2021, elect_2022


def count_topic_words(tokens, tf_idf_dict, topic_words):
    count = sum((tf_idf_dict[token] * 100) for token in tokens
                if token in topic_words)
    return count


def analyze_text_by_topics(tokens, tf_idf_dict):
    results = {
        'Health': count_topic_words(tokens, tf_idf_dict, health_words),
        'Crime': count_topic_words(tokens, tf_idf_dict, crime_words),
        'Education': count_topic_words(tokens, tf_idf_dict, education_words),
        'Defense': count_topic_words(tokens, tf_idf_dict, defense_words),
        'Foreign Relations': count_topic_words(tokens, tf_idf_dict,
                                               foreign_relations_words),
        'Religion': count_topic_words(tokens, tf_idf_dict, religion_words)
    }
    return results


def tokenize_text_from_articles(articles):
    tok = RegexpTokenizer(r"\b\w+(?:[`'’]\w+)?(?!'s)\b")
    exclusions = ["yesh atid-telem", "tikva hadasha", "yesh atid", "new hope", "blue and white", "derekh eretz", "joint list", "yisrael beiteinu", "united torah judaism"]
    text = ""
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
    return text


def get_tfidf_of_candidates(vectorizer, election_text, candidate_names):
    tf_idf_matrix = vectorizer.fit_transform([election_text])
    feature_names = vectorizer.get_feature_names_out()
    new_features = []
    for feature in feature_names:
        if "_" in feature:
            feature = feature.replace("_", " ")
        if "-" in feature:
            # yesh atid-telem
            feature = feature.split("-")[0]
        new_features.append(feature)
    dense_tfidf_matrix = tf_idf_matrix.todense()
    tf_idf_dict = {new_features[i]: dense_tfidf_matrix[0, i] for i in range(
        len(new_features))}
    return {a.lower(): tf_idf_dict[a.lower()] for a in candidate_names}


if __name__ == '__main__':
    # process = CrawlerProcess()
    # process.crawl(BodyTextSpider)
    # process.start()

    stop_words = set(stopwords.words('english'))
    articles_folders = ["elections2019a", "elections2019b", "elections2020",
                        "elections2021", "elections2022"]
    jerusalem_string = "jerusalem_article"
    times_string = "times_article"
    news_list = [jerusalem_string, times_string]

    all_articles = get_articles_list(articles_folders, news_list)
    jerusalem_articles = get_articles_list(articles_folders,
                                           [jerusalem_string])
    times_articles = get_articles_list(articles_folders,
                                       [times_string])

    (elect_2019a_article, elect_2019b_article, elect_2020_article,
     elect_2021_article, elect_2022_article) = (
        get_all_elections_list(articles_folders, news_list))

    # vectorizer = TfidfVectorizer(input='filename', stop_words='english', encoding='utf-8')
    # tfidf_matrix = vectorizer.fit_transform(articles)
    # feature_names = vectorizer.get_feature_names_out()
    # dense_tfidf_matrix = tfidf_matrix.todense()
    # print_sorted_tfidf(dense_tfidf_matrix, feature_names, articles)

    stemmer = porter.PorterStemmer()

    total_len = 0

    all_text = tokenize_text_from_articles(all_articles)
    jerusalem_text = tokenize_text_from_articles(jerusalem_articles)
    times_text = tokenize_text_from_articles(times_articles)
    elect_2019a_text = tokenize_text_from_articles(elect_2019a_article)
    elect_2019b_text = tokenize_text_from_articles(elect_2019b_article)
    elect_2020_text = tokenize_text_from_articles(elect_2020_article)
    elect_2021_text = tokenize_text_from_articles(elect_2021_article)
    elect_2022_text = tokenize_text_from_articles(elect_2022_article)

    vectorizer = TfidfVectorizer(token_pattern=r"\b\w+(?:[`'’]\w+)?(?!'s)\b",
                                 stop_words=list(stop_words), encoding='utf-8')

    tfidf_matrix = vectorizer.fit_transform([elect_2021_text])
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

    a = analyze_text_by_topics(new_features, tf_idf_dict)

    wordcloud = WordCloud(width=800, height=400, background_color='white'
                          ).generate_from_frequencies(tf_idf_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

    # party_dict = {name.lower(): count_topic_words(new_features, tf_idf_dict, candidate_names_2022) for name in candidate_names_2022_keys}
