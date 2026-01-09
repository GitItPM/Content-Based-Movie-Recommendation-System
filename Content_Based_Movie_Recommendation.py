import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class MovieRecommender:
    """
    Content-Based Movie Recommendation System
    Uses movie metadata to generate personalized recommendations
    """
    
    def __init__(self):
        self.movies_df = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.vectorizer = None
        
    def load_sample_data(self):
        """Load sample movie dataset"""
        movies_data = {
            'movie_id': range(1, 21),
            'title': [
                'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
                'Pulp Fiction', 'Fight Club', 'Forrest Gump', 'The Shawshank Redemption',
                'The Godfather', 'Goodfellas', 'Toy Story', 'Finding Nemo',
                'The Lion King', 'Frozen', 'Avengers: Endgame', 'Iron Man',
                'The Terminator', 'Blade Runner', 'Alien', 'Gravity'
            ],
            'genres': [
                'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi',
                'Action Sci-Fi', 'Crime Drama', 'Drama', 'Drama Romance',
                'Drama', 'Crime Drama', 'Crime Drama', 'Animation Comedy Family',
                'Animation Adventure Family', 'Animation Adventure Family',
                'Animation Adventure Family', 'Action Adventure Sci-Fi',
                'Action Adventure Sci-Fi', 'Action Sci-Fi Thriller',
                'Sci-Fi Thriller', 'Horror Sci-Fi', 'Drama Sci-Fi Thriller'
            ],
            'director': [
                'Christopher Nolan', 'Christopher Nolan', 'Christopher Nolan',
                'Wachowski Brothers', 'Quentin Tarantino', 'David Fincher',
                'Robert Zemeckis', 'Frank Darabont', 'Francis Ford Coppola',
                'Martin Scorsese', 'John Lasseter', 'Andrew Stanton',
                'Roger Allers', 'Chris Buck', 'Russo Brothers', 'Jon Favreau',
                'James Cameron', 'Ridley Scott', 'Ridley Scott', 'Alfonso Cuaron'
            ],
            'cast': [
                'Christian Bale Heath Ledger', 'Leonardo DiCaprio Marion Cotillard',
                'Matthew McConaughey Anne Hathaway', 'Keanu Reeves Laurence Fishburne',
                'John Travolta Samuel Jackson', 'Brad Pitt Edward Norton',
                'Tom Hanks Robin Wright', 'Tim Robbins Morgan Freeman',
                'Marlon Brando Al Pacino', 'Robert De Niro Ray Liotta',
                'Tom Hanks Tim Allen', 'Albert Brooks Ellen DeGeneres',
                'Matthew Broderick James Earl Jones', 'Kristen Bell Idina Menzel',
                'Robert Downey Chris Evans', 'Robert Downey Gwyneth Paltrow',
                'Arnold Schwarzenegger Linda Hamilton', 'Harrison Ford Rutger Hauer',
                'Sigourney Weaver Tom Skerritt', 'Sandra Bullock George Clooney'
            ],
            'description': [
                'Batman faces Joker in dark crime thriller',
                'Thieves enter dreams to plant ideas',
                'Astronauts search for new home in space',
                'Hacker discovers reality is simulation',
                'Intertwined stories of criminals in LA',
                'Insomniac forms underground fight club',
                'Man witnesses and influences American history',
                'Banker escapes prison after decades',
                'Mafia family struggles for power',
                'Rise and fall of mobster',
                'Toys come alive when humans leave',
                'Fish searches for lost son in ocean',
                'Lion cub becomes king of jungle',
                'Princess with ice powers saves kingdom',
                'Superheroes battle ultimate villain',
                'Billionaire builds powered armor suit',
                'Cyborg assassin from future hunts',
                'Blade runner hunts rogue androids',
                'Crew encounters deadly alien creature',
                'Astronauts stranded in space disaster'
            ],
            'year': [
                2008, 2010, 2014, 1999, 1994, 1999, 1994, 1994, 1972, 1990,
                1995, 2003, 1994, 2013, 2019, 2008, 1984, 1982, 1979, 2013
            ],
            'rating': [
                9.0, 8.8, 8.6, 8.7, 8.9, 8.8, 8.8, 9.3, 9.2, 8.7,
                8.3, 8.1, 8.5, 7.4, 8.4, 7.9, 8.0, 8.1, 8.4, 7.7
            ]
        }
        
        self.movies_df = pd.DataFrame(movies_data)
        print("Sample movie dataset loaded successfully")
        print(f"  Total movies: {len(self.movies_df)}\n")
        
    def create_content_features(self, weights: Dict[str, float] = None):
        """
        Feature Extraction: Create combined feature representation
        
        Args:
            weights: Dictionary with feature importance weights
        """
        if weights is None:
            weights = {
                'genres': 3.0,
                'director': 2.0,
                'cast': 1.5,
                'description': 1.0
            }
        
        print("=== Feature Extraction Process ===")
        print(f"Feature weights: {weights}\n")
        
        # Create weighted content string for each movie
        self.movies_df['content'] = (
            (self.movies_df['genres'] + ' ') * int(weights['genres']) +
            (self.movies_df['director'] + ' ') * int(weights['director']) +
            (self.movies_df['cast'] + ' ') * int(weights['cast']) +
            self.movies_df['description'] * int(weights['description'])
        )
        
        print("Content features created")
        print(f"  Sample content string:")
        print(f"  '{self.movies_df['content'].iloc[0][:100]}...'\n")
        
    def vectorize_features(self, method='tfidf', max_features=500):
        """
        Feature Vectorization: Convert text to numerical vectors
        
        Args:
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
        """
        print("=== Feature Vectorization ===")
        print(f"Method: {method.upper()}")
        print(f"Max features: {max_features}\n")
        
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        # Transform content to feature matrix
        self.feature_matrix = self.vectorizer.fit_transform(
            self.movies_df['content']
        )
        
        print(f"Feature matrix created")
        print(f"  Shape: {self.feature_matrix.shape}")
        print(f"  (movies × features)\n")
        
        # Show top features
        feature_names = self.vectorizer.get_feature_names_out()
        print(f"  Sample features: {list(feature_names[:10])}\n")
        
    def compute_similarity(self):
        """Compute cosine similarity between all movies"""
        print("=== Computing Similarity Matrix ===")
        print("Using Cosine Similarity metric\n")
        
        # Compute pairwise cosine similarity
        self.similarity_matrix = cosine_similarity(
            self.feature_matrix,
            self.feature_matrix
        )
        
        print(f"Similarity matrix computed")
        print(f"  Shape: {self.similarity_matrix.shape}")
        print(f"  (movies × movies)\n")
        
        # Show similarity statistics
        np.fill_diagonal(self.similarity_matrix, 0)  # Exclude self-similarity
        avg_sim = np.mean(self.similarity_matrix)
        max_sim = np.max(self.similarity_matrix)
        
        print(f"  Average similarity: {avg_sim:.4f}")
        print(f"  Maximum similarity: {max_sim:.4f}\n")
        
    def get_recommendations(
        self,
        movie_title: str,
        n_recommendations: int = 5,
        min_rating: float = 0.0
    ) -> pd.DataFrame:
        """
        Generate recommendations for a given movie
        
        Args:
            movie_title: Title of the movie to base recommendations on
            n_recommendations: Number of recommendations to return
            min_rating: Minimum rating filter
            
        Returns:
            DataFrame with recommended movies and similarity scores
        """
        # Find movie index
        movie_matches = self.movies_df[
            self.movies_df['title'].str.contains(movie_title, case=False)
        ]
        
        if movie_matches.empty:
            print(f"Movie '{movie_title}' not found!")
            return pd.DataFrame()
        
        movie_idx = movie_matches.index[0]
        movie_name = self.movies_df.iloc[movie_idx]['title']
        
        print(f"\n{'='*60}")
        print(f"Generating Recommendations for: '{movie_name}'")
        print(f"{'='*60}\n")
        
        # Get similarity scores for this movie
        similarity_scores = list(enumerate(self.similarity_matrix[movie_idx]))
        
        # Sort by similarity score (descending)
        similarity_scores = sorted(
            similarity_scores,
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter and collect recommendations
        recommendations = []
        for idx, score in similarity_scores:
            if idx == movie_idx:  # Skip the movie itself
                continue
            
            movie = self.movies_df.iloc[idx]
            if movie['rating'] >= min_rating:
                recommendations.append({
                    'rank': len(recommendations) + 1,
                    'title': movie['title'],
                    'similarity_score': score,
                    'genres': movie['genres'],
                    'director': movie['director'],
                    'year': movie['year'],
                    'rating': movie['rating']
                })
            
            if len(recommendations) >= n_recommendations:
                break
        
        return pd.DataFrame(recommendations)
    
    def display_recommendations(self, recommendations: pd.DataFrame):
        """Display recommendations in formatted output"""
        if recommendations.empty:
            return
        
        print("Recommended Movies:")
        print("-" * 60)
        
        for _, movie in recommendations.iterrows():
            print(f"\n{int(movie['rank'])}. {movie['title']} ({int(movie['year'])})")
            print(f"   Similarity Score: {movie['similarity_score']:.4f}")
            print(f"   Genres: {movie['genres']}")
            print(f"   Director: {movie['director']}")
            print(f"   Rating: {movie['rating']}/10")
        
        print("\n" + "=" * 60 + "\n")
    
    def analyze_recommendation(self, original_title: str, recommended_title: str):
        """Detailed analysis of why a movie was recommended"""
        orig_idx = self.movies_df[
            self.movies_df['title'] == original_title
        ].index[0]
        rec_idx = self.movies_df[
            self.movies_df['title'] == recommended_title
        ].index[0]
        
        orig_movie = self.movies_df.iloc[orig_idx]
        rec_movie = self.movies_df.iloc[rec_idx]
        
        print(f"\n{'='*60}")
        print(f"Recommendation Analysis")
        print(f"{'='*60}\n")
        print(f"Original: {orig_movie['title']}")
        print(f"Recommended: {rec_movie['title']}")
        print(f"\nSimilarity Score: {self.similarity_matrix[orig_idx][rec_idx]:.4f}\n")
        
        print("Shared Attributes:")
        print("-" * 40)
        
        # Genre overlap
        orig_genres = set(orig_movie['genres'].split())
        rec_genres = set(rec_movie['genres'].split())
        common_genres = orig_genres & rec_genres
        if common_genres:
            print(f"Genres: {', '.join(common_genres)}")
        
        # Director match
        if orig_movie['director'] == rec_movie['director']:
            print(f"Director: {orig_movie['director']}")
        
        # Cast overlap
        orig_cast = set(orig_movie['cast'].split())
        rec_cast = set(rec_movie['cast'].split())
        common_cast = orig_cast & rec_cast
        if common_cast:
            print(f"Cast: {', '.join(common_cast)}")
        
        print("\n" + "=" * 60 + "\n")
    
    def get_statistics(self):
        """Display system statistics"""
        print("\n" + "="*60)
        print("RECOMMENDATION SYSTEM STATISTICS")
        print("="*60 + "\n")
        
        print(f"Total Movies: {len(self.movies_df)}")
        print(f"Feature Vector Dimensions: {self.feature_matrix.shape[1]}")
        print(f"Total Similarity Scores: {self.similarity_matrix.size}")
        
        print(f"\nGenre Distribution:")
        all_genres = ' '.join(self.movies_df['genres']).split()
        genre_counts = pd.Series(all_genres).value_counts()
        for genre, count in genre_counts.head(5).items():
            print(f"  {genre}: {count} movies")
        
        print(f"\nRating Distribution:")
        print(f"  Average: {self.movies_df['rating'].mean():.2f}")
        print(f"  Highest: {self.movies_df['rating'].max():.2f}")
        print(f"  Lowest: {self.movies_df['rating'].min():.2f}")
        
        print("\n" + "="*60 + "\n")


def add_custom_movie(recommender):
    """Allow user to add a custom movie to the dataset"""
    print("\n" + "="*60)
    print("ADD CUSTOM MOVIE")
    print("="*60 + "\n")
    
    title = input("Enter movie title: ").strip()
    genres = input("Enter genres (space-separated, e.g., 'Action Sci-Fi'): ").strip()
    director = input("Enter director name: ").strip()
    cast = input("Enter main cast (space-separated names): ").strip()
    description = input("Enter brief description: ").strip()
    year = input("Enter release year: ").strip()
    rating = input("Enter rating (0-10): ").strip()
    
    try:
        year = int(year)
        rating = float(rating)
        
        new_movie = {
            'movie_id': len(recommender.movies_df) + 1,
            'title': title,
            'genres': genres,
            'director': director,
            'cast': cast,
            'description': description,
            'year': year,
            'rating': rating
        }
        
        recommender.movies_df = pd.concat([
            recommender.movies_df,
            pd.DataFrame([new_movie])
        ], ignore_index=True)
        
        print(f"\nMovie '{title}' added successfully!")
        print("  Note: Rebuild the model to include this movie in recommendations.\n")
        
    except ValueError:
        print("\nInvalid input. Movie not added.\n")


def display_all_movies(recommender):
    """Display all movies in the dataset"""
    print("\n" + "="*60)
    print("ALL MOVIES IN DATABASE")
    print("="*60 + "\n")
    
    for idx, movie in recommender.movies_df.iterrows():
        print(f"{idx + 1}. {movie['title']} ({int(movie['year'])})")
        print(f"   Genres: {movie['genres']}")
        print(f"   Director: {movie['director']}")
        print(f"   Rating: {movie['rating']}/10\n")
    
    print(f"Total: {len(recommender.movies_df)} movies\n")


def search_movies(recommender):
    """Search for movies by title, genre, or director"""
    print("\n" + "="*60)
    print("SEARCH MOVIES")
    print("="*60 + "\n")
    
    print("Search by:")
    print("1. Title")
    print("2. Genre")
    print("3. Director")
    choice = input("\nEnter choice (1-3): ").strip()
    
    search_term = input("Enter search term: ").strip().lower()
    
    if choice == '1':
        results = recommender.movies_df[
            recommender.movies_df['title'].str.lower().str.contains(search_term)
        ]
    elif choice == '2':
        results = recommender.movies_df[
            recommender.movies_df['genres'].str.lower().str.contains(search_term)
        ]
    elif choice == '3':
        results = recommender.movies_df[
            recommender.movies_df['director'].str.lower().str.contains(search_term)
        ]
    else:
        print("Invalid choice!")
        return
    
    if results.empty:
        print(f"\nNo movies found matching '{search_term}'\n")
    else:
        print(f"\nFound {len(results)} movie(s):\n")
        for _, movie in results.iterrows():
            print(f"• {movie['title']} ({int(movie['year'])})")
            print(f"  Genres: {movie['genres']}")
            print(f"  Director: {movie['director']}")
            print(f"  Rating: {movie['rating']}/10\n")


def interactive_recommendations(recommender):
    """Interactive recommendation interface"""
    print("\n" + "="*60)
    print("GET MOVIE RECOMMENDATIONS")
    print("="*60 + "\n")
    
    # Show available movies
    print("Available movies:")
    for idx, title in enumerate(recommender.movies_df['title'], 1):
        print(f"{idx}. {title}")
    
    movie_input = input("\nEnter movie title (or part of it): ").strip()
    
    # Find matching movies
    matches = recommender.movies_df[
        recommender.movies_df['title'].str.contains(movie_input, case=False)
    ]
    
    if matches.empty:
        print(f"\nNo movie found matching '{movie_input}'")
        return
    
    if len(matches) > 1:
        print(f"\nMultiple matches found:")
        for idx, movie in enumerate(matches['title'], 1):
            print(f"{idx}. {movie}")
        choice = input(f"\nSelect movie (1-{len(matches)}): ").strip()
        try:
            selected_movie = matches.iloc[int(choice) - 1]['title']
        except:
            print("Invalid selection!")
            return
    else:
        selected_movie = matches.iloc[0]['title']
    
    # Get parameters
    try:
        n_recs = input("\nNumber of recommendations (default 5): ").strip()
        n_recs = int(n_recs) if n_recs else 5
        
        min_rating = input("Minimum rating filter (0-10, default 0): ").strip()
        min_rating = float(min_rating) if min_rating else 0.0
        
        # Generate recommendations
        recommendations = recommender.get_recommendations(
            selected_movie,
            n_recommendations=n_recs,
            min_rating=min_rating
        )
        
        recommender.display_recommendations(recommendations)
        
        # Ask if user wants detailed analysis
        if not recommendations.empty:
            analyze = input("Would you like detailed analysis for any recommendation? (y/n): ").strip().lower()
            if analyze == 'y':
                rec_num = input(f"Enter recommendation number (1-{len(recommendations)}): ").strip()
                try:
                    rec_title = recommendations.iloc[int(rec_num) - 1]['title']
                    recommender.analyze_recommendation(selected_movie, rec_title)
                except:
                    print("Invalid selection!")
        
    except ValueError:
        print("\nInvalid input!")


def build_model(recommender):
    """Build/rebuild the recommendation model"""
    print("\n" + "="*60)
    print("BUILDING RECOMMENDATION MODEL")
    print("="*60 + "\n")
    
    print("Step 1: Feature Extraction...")
    recommender.create_content_features()
    
    print("Step 2: Feature Vectorization...")
    recommender.vectorize_features(method='tfidf')
    
    print("Step 3: Computing Similarity Matrix...")
    recommender.compute_similarity()
    
    print("\nModel built successfully!\n")


def demo_mode(recommender):
    """Run automatic demonstration"""
    print("\n" + "="*60)
    print("RUNNING DEMONSTRATION MODE")
    print("="*60 + "\n")
    
    # Build model
    print("Building model...")
    build_model(recommender)
    
    # Example recommendations
    print("\n--- Example 1: Sci-Fi Thriller ---")
    recs1 = recommender.get_recommendations('Inception', n_recommendations=5)
    recommender.display_recommendations(recs1)
    
    print("\n--- Example 2: Animated Family ---")
    recs2 = recommender.get_recommendations('Lion King', n_recommendations=5)
    recommender.display_recommendations(recs2)
    
    print("\n--- Example 3: High-Rated Crime Drama ---")
    recs3 = recommender.get_recommendations('Godfather', n_recommendations=5, min_rating=8.0)
    recommender.display_recommendations(recs3)
    
    print("\n--- Detailed Analysis ---")
    recommender.analyze_recommendation('Inception', 'Interstellar')
    
    recommender.get_statistics()


def main():
    """Main interactive function"""
    print("\n" + "╔" + "="*58 + "╗")
    print("║  CONTENT-BASED MOVIE RECOMMENDATION SYSTEM - PYTHON  ║")
    print("║  Machine Learning & Recommendation Systems          ║")
    print("╚" + "="*58 + "╝\n")
    
    # Initialize recommender
    recommender = MovieRecommender()
    recommender.load_sample_data()
    
    model_built = False
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("\n1.  Build/Rebuild Recommendation Model")
        print("2.  Get Movie Recommendations")
        print("3.  Add Custom Movie")
        print("4.  View All Movies")
        print("5.  Search Movies")
        print("6.  View System Statistics")
        print("7.  Run Demo Mode")
        print("8.  Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            build_model(recommender)
            model_built = True
            
        elif choice == '2':
            if not model_built:
                print("\nModel not built yet! Building now...")
                build_model(recommender)
                model_built = True
            interactive_recommendations(recommender)
            
        elif choice == '3':
            add_custom_movie(recommender)
            if model_built:
                print("Remember to rebuild the model to include your movie!")
            
        elif choice == '4':
            display_all_movies(recommender)
            
        elif choice == '5':
            search_movies(recommender)
            
        elif choice == '6':
            if not model_built:
                print("\nModel not built yet! Build it first to see statistics.")
            else:
                recommender.get_statistics()
            
        elif choice == '7':
            demo_mode(recommender)
            model_built = True
            
        elif choice == '8':
            print("\n" + "="*60)
            print("Thank you for using the Movie Recommendation System!")
            print("="*60 + "\n")
            print("Key Concepts Demonstrated:")
            print("  • Feature extraction and vectorization")
            print("  • TF-IDF text representation")
            print("  • Cosine similarity computation")
            print("  • Similarity-based ranking")
            print("  • Personalized recommendation generation\n")
            break
            
        else:
            print("\nInvalid choice! Please enter a number between 1-8.")


if __name__ == "__main__":
    main()