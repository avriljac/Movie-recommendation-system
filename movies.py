
import spacy
from sklearn.metrics.pairwise import cosine_similarity


# Load spaCy model
nlp = spacy.load('en_core_web_md')

# Read movie descriptions from the file
def read_movie_descriptions(file_path='movies.txt'):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Process the descriptions using spaCy
def process_descriptions(descriptions):
    return [nlp(description) for description in descriptions]

# Get the most similar movie title based on a given description
def get_most_similar_movie(description, processed_descriptions, titles):
    if not processed_descriptions or not any(processed_descriptions):
        return "No movies available"

    description_vector = nlp(description).vector
    similarity_scores = [cosine_similarity([description_vector], [doc.vector])[0][0] for doc in processed_descriptions]

    if not any(similarity_scores):
        return "No movies available"

    most_similar_index = similarity_scores.index(max(similarity_scores))
    return titles[most_similar_index]

# Function to find the next movie based on a watched movie description
def find_next_movie(watched_description, file_path='movies.txt'):
    # Read movie descriptions and titles
    descriptions = read_movie_descriptions(file_path)
    titles = [line.split(':')[0].strip() for line in descriptions]

    # Process the movie descriptions using spaCy
    processed_descriptions = process_descriptions(descriptions)

    # Get the most similar movie title
    next_movie = get_most_similar_movie(watched_description, processed_descriptions, titles)

    return next_movie

# Example usage
if __name__ == "__main__":
    watched_description = "Will he save their world or destroy it? When the Hulk becomes too dangerous for the Earth, the Illuminati trick Hulk into a shuttle and launch him into space to a planet where the Hulk can live in peace. Unfortunately, Hulk lands on the planet Sakaar where he is sold into slavery and trained as a gladiator."
    next_movie = find_next_movie(watched_description)
    print(f"The next movie to watch is: {next_movie}")