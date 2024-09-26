import numpy as np
import nltk as nl
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import scrapy
from nltk.stem import porter
from wordcloud import WordCloud
from textblob import TextBlob
from summa import summarizer
import os
import textstat
import matplotlib.pyplot as plt


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

US_election_words = [
    'biden', 'trump', 'republican', 'democrat', 'harris', 'pence'
]

candidate_names_2022 = {
    "Likud": ["netanyahu", "bibi", "likud"],
    "Yesh Atid": ["lapid", "yesh atid"],
    "RZP–Otzma": ["rzp", "otzma", "religious zionist party", "smotrich"],
    "National Unity": ["gantz", "national unity"],
    "Shas": ["shas", "deri"]

}
candidate_names_2021 = {
    "Likud": ["netanyahu", "bibi", "likud"],
    "Yesh Atid": ["lapid", "yesh atid"],
    "Shas": ["shas", "deri"],
    "Blue and White": ["blue and white", "gantz"],
    "Yamina": ["bennett", "yamina"]

}

candidate_names_2020 = {
    "Likud": ["netanyahu", "bibi", "likud"],
    "Blue and White": ["blue and white", "gantz"],
    "Joint List": ["joint list", "odeh"],
    "Shas": ["shas", "deri"],
    "Gimel": ["utj", "gimel","united torah judaism", "litzman"]

}



candidate_names_2019b = {
    "Blue and White": ["blue and white", "gantz"],
    "Likud": ["netanyahu", "bibi", "likud"],
    "Joint List": ["joint list", "odeh"],
    "Shas": ["shas", "deri"],
    "Yisrael Beytenu": ["yisrael beytenu", "liberman"]

}

candidate_names_2019a = {
    "Likud": ["netanyahu", "bibi", "likud"],
    "Blue and White": ["blue and white", "gantz"],
    "Shas": ["shas", "deri"],
    "Gimel": ["utj", "gimel","united torah judaism", "litzman"],
    "Labor": ["labor", "gabbay"]

}



seat_count_2022 = {
    "Likud": 32,
    "Yesh Atid": 24,
    "RZP–Otzma": 14,
    "National Unity": 12,
    "Shas": 11

}



seat_count_2021 = {
    "Likud": 30,
    "Yesh Atid": 17,
    "Shas": 9,
    "Blue and White": 8,
    "Yamina": 7

}
seat_count_2020 = {
    "Likud": 36,
    "Blue and White": 33,
    "Joint List": 15,
    "Shas": 9,
    "Gimel": 7

}

seat_count_2019b = {
    "Blue and White": 33,
    "Likud": 32,
    "Joint List": 13,
    "Shas": 9,
    "Yisrael Beytenu": 8

}

seat_count_2019a = {
    "Likud": 35,
    "Blue and White": 35,
    "Shas":8,
    "Gimel": 8,
    "Labor": 6

}

save_path = 'results/'

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




def get_tfidf_score_and_indices(dense_matrix):
    tfidf_scores = dense_matrix[0].tolist()[0]
    tfidf_scores = np.array(tfidf_scores)
    sorted_indices = np.argsort(tfidf_scores)[::-1]
    return tfidf_scores, sorted_indices


def print_sorted_tfidf(dense_matrix, feature_names, filename):
    print(f"TF-IDF values for {filename}:")
    tfidf_scores, sorted_indices = get_tfidf_score_and_indices(dense_matrix)

    for idx in sorted_indices:
        if tfidf_scores[idx] > 0:  # Print only non-zero values
            print(f"{feature_names[idx]}: {tfidf_scores[idx]:.4f}")
    print("\n")


def plot_reality_check(dense_matrix, feature_names, top_n_words, plot_file_name):
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
    plt.savefig(save_path + plot_file_name)
    plt.close()


def get_articles_list(folders, news_lst):
    articles_lst = []
    for folder in folders:
        range_articles = 11
        articles_lst += [
            f'{folder}\\{name}{i}.txt'
            for name in news_lst
            for i in range(1, range_articles)
        ]
    return articles_lst


def get_election_list(folder, news_lst):
    election_lst = []
    range_articles = 11
    election_lst += [
        f'{folder}\\{name}{i}.txt'
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
        'Religion': count_topic_words(tokens, tf_idf_dict, religion_words),
        'US Election': count_topic_words(tokens, tf_idf_dict, US_election_words)
    }
    return results


def tokenize_text_from_articles(articles):
    tok = RegexpTokenizer(r"\b\w+(?:[`'’]\w+)?(?!'s)\b")
    space_exclusions = ["yisrael beytenu","united torah judaism","national unity",  "tikva hadasha", "religious zionist party", "yesh atid", "new hope", "blue and white", "derekh eretz", "joint list", "yisrael beiteinu", "united torah judaism"]
    apostrophe_exclusions = ["ya'alon","sa'ar"]
    text = ""
    for article in articles:
        with open(article, "r", encoding='utf-8') as f:
            a = f.read().lower()

            for exc in space_exclusions:
                a = a.replace(exc, "_".join(exc.split(" ")))
            for exc in apostrophe_exclusions:
                a = a.replace(exc, "".join(exc.split("'")))

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


def count_candidate_words(tokens, tf_idf_dict, candidate_words):
    # Initialize a dictionary to store the TF-IDF sums for each candidate
    candidate_tfidf_sums = {}

    for candidate, words in candidate_words.items():
        # Calculate the sum of TF-IDF scores for the candidate's words
        total_tfidf = sum((tf_idf_dict.get(token, 0) * 100) for token in tokens if token in words)
        candidate_tfidf_sums[candidate] = total_tfidf

    return candidate_tfidf_sums

def analyze_election_year(text, year, candidate_parties, seat_counts):
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

    plot_reality_check(dense_tfidf_matrix, new_features, 30, f'top_features{year}')
    tf_idf_dict = {new_features[i]: dense_tfidf_matrix[0, i] for i in range(
        len(new_features))}

    res = analyze_text_by_topics(new_features, tf_idf_dict)
    party_scores = count_candidate_words(new_features, tf_idf_dict, candidate_parties)
    plot_candidate_scores(party_scores, year, seat_counts)

    wordcloud = WordCloud(width=800, height=400, background_color='white'
                          ).generate_from_frequencies(tf_idf_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(save_path + f'word_cloud_{year}.png')
    plt.close()
    return res
def plot_candidate_scores(party_scores, year, seat_counts):
    # Extract candidates and their scores
    parties = list(party_scores.keys())
    scores = [float(score) for score in party_scores.values()]  # Convert np.float64 to float

    # Create x-axis labels with seat counts, only the first party will say "seats"
    party_labels = [
        f"{party} ({seat_counts.get(party, 0)}{' seats' if i == 0 else ''})"
        for i, party in enumerate(parties)
    ]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(party_labels, scores, align='center')
    plt.xlabel('Parties')
    plt.ylabel('TF-IDF Score')
    plt.title(f'TF-IDF Scores for Candidate Parties in {year}')
    plt.xticks(rotation=45, ha='right')
    plt.ylim((0, 70))
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(save_path + f'party_scores_{year}.png')
    plt.close()  # Close the plot to free up memory
def plot_year_comparison(year_data, features):


    for feature in features:
        # Extract values for the feature from each year's data
        values = [year_data[year].get(feature, np.nan) for year in sorted(year_data.keys())]

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(year_data.keys(), values, align='center')
        plt.xlabel('Year')
        plt.ylabel('Value')
        plt.title(f'Comparison of {feature} Over Years')
        plt.ylim((0,50))  # Set y-axis limits
        plt.tight_layout()

        # Save the figure to a file
        plt.savefig(save_path + f'{feature}_comparison.png')
        plt.close()  # Close the plot to free up memory

def plot_averages(year_data):

    parameters = list(year_data['year_2019a'].keys())  # Assuming each year has the same parameters

    averages = []
    variances = []

    for param in parameters:
        param_values = [year_data[year][param] for year in year_data]

        avg = np.mean(param_values)
        var = np.var(param_values)

        averages.append(avg)
        variances.append(var)

    plt.figure(figsize=(12, 6))
    bar_width = 0.35
    index = np.arange(len(parameters))

    plt.bar(index, averages, bar_width, label='Average', color='blue')

    plt.bar(index + bar_width, variances, bar_width, label='Variance', color='orange')

    plt.xlabel('Parameters')
    plt.ylabel('Value')
    plt.title('Average and Variance of Election Parameters')
    plt.xticks(index + bar_width / 2, parameters, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path + 'average_variance_election_parameters.png')






def summarize_articles(year):
    summaries = {}
    year_folder = f"elections{year}"
    folder_path = os.path.join(year_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                summary = summarizer.summarize(content, ratio=0.1)
                summaries[filename] = summary


    return summaries


def summarize_by_year(election_year):
    summaries = summarize_articles(election_year)

    # Save summaries to a file
    with open(save_path + f'{election_year}_summaries.txt', 'w', encoding='utf8') as f:
        for filename, summary in summaries.items():
            f.write(f"\nSummary for {filename}:\n")
            f.write(summary)
            f.write("\n\n")

def summarize_years(years):
    for year in years:
        summarize_by_year(year)


def perform_sentiment_analysis(text):

    blob = TextBlob(text)
    return blob.sentiment.polarity


def analyze_sentiment_for_folder(year):

    sentiment_scores = {}

    for filename in os.listdir(f'elections{year}'):
        file_path = os.path.join(f'elections{year}', filename)

        # Open and read the content of each file
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Perform sentiment analysis
        polarity_score = perform_sentiment_analysis(content)
        sentiment_scores[filename] = polarity_score

    return sentiment_scores


def plot_sentiment_scores(sentiment_scores, year):

    files = list(sentiment_scores.keys())
    scores = list(sentiment_scores.values())

    plt.figure(figsize=(10, 6))
    plt.bar(files, scores, color='blue')
    plt.xlabel('Files')
    plt.ylabel('Polarity Score')
    plt.title(f'Sentiment Polarity Scores for {year} Elections')
    plt.xticks(rotation=45, ha='right')
    plt.ylim((-0.2,0.2))
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(save_path + f'sentiment_polarity_{year}.png')

def create_sentiment_plots(years):
    for year in years:
        sentiment = analyze_sentiment_for_folder(year)
        plot_sentiment_scores(sentiment, year)






def analyze_readability(text):
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    flesch_kincaid_grade = textstat.flesch_kincaid_grade(text)
    smog_index = textstat.smog_index(text)

    return {
        "Flesch Reading Ease": flesch_reading_ease,
        "Flesch-Kincaid Grade": flesch_kincaid_grade,
        "SMOG Index": smog_index
    }


def analyze_year_folder(year):
    readability_scores = []

    # Read all text files in the specified folder
    for filename in os.listdir(f'elections{year}'):
        if filename.endswith(".txt"):  # Assuming text files have .txt extension
            with open(os.path.join(f'elections{year}', filename), 'r', encoding='utf-8') as file:
                text = file.read()
                scores = analyze_readability(text)
                readability_scores.append(scores)

    # Calculate average readability scores
    if readability_scores:
        avg_scores = {
            "Flesch Reading Ease": np.mean([score["Flesch Reading Ease"] for score in readability_scores]),
            "Flesch-Kincaid Grade": np.mean([score["Flesch-Kincaid Grade"] for score in readability_scores]),
            "SMOG Index": np.mean([score["SMOG Index"] for score in readability_scores])
        }
    else:
        avg_scores = {
            "Flesch Reading Ease": 0,
            "Flesch-Kincaid Grade": 0,
            "SMOG Index": 0
        }

    return avg_scores


def analyze_readability_across_years(years):
    results = {}
    for year in years:

        results[year] = analyze_year_folder(year)
    return results


def plot_readability_results(readability_results):
    metrics = list(readability_results[next(iter(readability_results))].keys())
    x = list(readability_results.keys())

    for metric in metrics:
        y = [readability_results[year][metric] for year in x]

        plt.figure(figsize=(10, 6))
        plt.bar(x, y)
        plt.xlabel('Years')
        plt.ylabel(metric)
        plt.title(f'Average {metric} Across Election Years')
        plt.tight_layout()
        plt.savefig(save_path + f'{metric.lower().replace(" ", "_")}_comparison.png')



if __name__ == '__main__':


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

    year_data ={
    'year_2019a' : analyze_election_year(elect_2019a_text, '2019a', candidate_names_2019a, seat_count_2019a),
    'year_2019b' : analyze_election_year(elect_2019b_text, '2019b', candidate_names_2019b, seat_count_2019b),
    'year_2020' : analyze_election_year(elect_2020_text, '2020', candidate_names_2020, seat_count_2020),
    'year_2021' : analyze_election_year(elect_2021_text, '2021', candidate_names_2021,seat_count_2021),
    'year_2022' : analyze_election_year(elect_2022_text, '2022', candidate_names_2022, seat_count_2022)   }
    features = set()
    for year in year_data:
        features.update(year_data[year].keys())

    # Generate comparison plots for each feature
    plot_year_comparison(year_data, features)
    plot_averages(year_data)
    years = ['2022', '2021', '2020', '2019b', '2019a']
    summarize_years(years)
    create_sentiment_plots(years)
    plot_readability_results(analyze_readability_across_years(years))





