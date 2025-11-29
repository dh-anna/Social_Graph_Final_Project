import pandas as pd
import numpy as np
from typing import Dict, List, Set
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class Collaboration:
    year: float
    film: str
    director: str
    film_id: str


@dataclass
class ActorCareerAnalysis:
    actor: str
    career_start: float
    career_end: float
    career_length: float
    total_films: int
    num_collabs: int
    avg_collab_year: float
    career_position: float
    mean_career_phase: str
    actor_phase_summary: str
    collabs: List[Collaboration]


class ActorDirectorAnalyzer:
    def __init__(self, cluster_to_nodes: Dict[int, Set[str]], name_lookup: Dict[str, str], min_films:int=5, min_career_length:int=5, popularity_percentile:float=0.75):
        self.df_actors = None
        self.df_movie_directors = None
        self.df_celebrity = None
        self.current_cluster_id = None
        self.cluster_to_nodes = cluster_to_nodes
        self.name_lookup = name_lookup
        self.min_films = min_films
        self.min_career_length = min_career_length
        self.popularity_percentile = popularity_percentile
        self.EARLY_CAREER = 0.33
        self.LATE_CAREER = 0.66

    def load_data(self, df_celebrity:pd.DataFrame, df_actors:pd.DataFrame, df_movie_directors:pd.DataFrame, cluster_popular_directors:Dict[int, List[str]]) :
        # Clean data
        df_actors['Year'] = pd.to_numeric(self.df_actors['Year'], errors='coerce')
        df_actors = self.df_actors.dropna(subset=['Year', 'Actor', 'FilmID'])
        return df_celebrity, df_actors, df_movie_directors, cluster_popular_directors

    def get_popular_actors(self, df_celebrity: pd.DataFrame)-> Set[str]:
        df_actors_celebrity = df_celebrity[df_celebrity['known_for_department'] == 'Acting'].copy()

        if df_actors_celebrity.empty:
            raise ValueError("No actors found in celebrity dataset")

        # Calculate popularity threshold
        popularity_threshold = df_actors_celebrity['popularity'].quantile(self.popularity_percentile)

        # Get popular actors
        popular_actors = set(
            df_actors_celebrity[df_actors_celebrity['popularity'] >= popularity_threshold]['name']
        )

        return popular_actors

    def create_director_lookup(self, df_movie_directors:pd.DataFrame)-> Dict:
        director_lookup = {}

        for _, row in df_movie_directors.iterrows():
            film_id = row['tconst']
            directors = row['directors']

            if pd.notna(directors):
                director_ids = directors.split(',')
                director_names = set()

                for dir_id in director_ids:
                    if dir_id in self.name_lookup:
                        director_names.add(self.name_lookup[dir_id])

                if director_names:
                    director_lookup[film_id] = director_names

        return director_lookup

    def analyze_actor_career(self, actor:str, df_actors:pd.DataFrame, director_lookup:Dict, cluster_directors, popular_directors )->ActorCareerAnalysis:
        # Get actor's filmography
        actor_films = df_actors[df_actors['Actor'] == actor].sort_values('Year')

        # Check minimum requirements
        if len(actor_films) < self.min_films:
            return None

        # Calculate career metrics
        career_start = actor_films['Year'].min()
        career_end = actor_films['Year'].max()
        career_length = career_end - career_start

        if career_length < self.min_career_length:
            return None

        # Find collaborations with cluster directors
        collaborations = []
        popular_directors = set(popular_directors)

        actor_phases = set()


        for _, film in actor_films.iterrows():
            film_id = film['FilmID']

            if film_id in director_lookup:
                film_directors = director_lookup[film_id]
                # Ensure both are sets before intersection
                if not isinstance(cluster_directors, set):
                    cluster_directors = set(cluster_directors)
                cluster_directors_in_film = film_directors & cluster_directors & popular_directors

                for director in cluster_directors_in_film:
                    collaborations.append(Collaboration(
                        year=float(film['Year']),
                        film=film['Film'],
                        director=director,
                        film_id=film_id
                    ))

                    year = float(film['Year'])
                    position = (year - career_start) / career_length
                    if position < self.EARLY_CAREER:
                        actor_phases.add('early')
                    elif position > self.LATE_CAREER:
                        actor_phases.add('late')
                    else:
                        actor_phases.add('middle')

        if not collaborations:
            return None

        # Calculate collaboration metrics
        collab_years = [c.year for c in collaborations]
        avg_collab_year = float(np.mean(collab_years))

        # Calculate career position (0 = start, 1 = end) - ensure float
        if career_length > 0:
            career_position = float((avg_collab_year - career_start) / career_length)
        else:
            career_position = 0.0

        # Determine career phase
        if career_position < self.EARLY_CAREER:
            career_phase = "Early"
        elif career_position > self.LATE_CAREER:
            career_phase = "Late"
        else:
            career_phase = "Middle"

        actor_phase_summary = ""

        if actor_phases:
            # Categorize by combination
            if actor_phases == {'early', 'middle', 'late'}:
                actor_phase_summary = "all_phases"
            elif actor_phases == {'early'}:
                actor_phase_summary = "early_only"
            elif actor_phases == {'middle'}:
                actor_phase_summary = "middle_only"
            elif actor_phases == {'late'}:
                actor_phase_summary = "late_only"
            elif actor_phases == {'early', 'middle'}:
                actor_phase_summary = "early_middle"
            elif actor_phases == {'middle', 'late'}:
                actor_phase_summary = "middle_late"
            elif actor_phases == {'early', 'late'}:
                actor_phase_summary = "early_late"
        else:
            actor_phase_summary = "never"

        return ActorCareerAnalysis(
            actor=actor,
            career_start=career_start,
            career_end=career_end,
            career_length=career_length,
            total_films=len(actor_films),
            num_collabs=len(collaborations),
            avg_collab_year=avg_collab_year,
            career_position=career_position,
            mean_career_phase=career_phase,
            actor_phase_summary = actor_phase_summary,
            collabs=collaborations
        )

    def analyze_all_actors(self, cluster_id:int)->List[ActorCareerAnalysis]:
        # Store cluster_id for use in reporting

        df_celebrity, df_actors, df_movie_directors, cluster_popular_directors = self.load_data()
        self.current_cluster_id = cluster_id

        # Get cluster directors
        if cluster_id not in self.cluster_to_nodes:
            raise ValueError(f"Cluster {cluster_id} not found in cluster_to_nodes")

        cluster_directors = set(self.cluster_to_nodes[cluster_id])

        # Get popular actors
        popular_actors = self.get_popular_actors(df_celebrity)

        # Create director lookup for efficiency
        director_lookup = self.create_director_lookup(df_movie_directors)

        results = []
        actors_processed = 0

        for actor in popular_actors:
            analysis = self.analyze_actor_career(
                actor, df_actors, director_lookup, cluster_directors, cluster_popular_directors[cluster_id]
            )

            if analysis:
                results.append(analysis)

            actors_processed += 1

        # Sort by career position - ensure float conversion to avoid numpy boolean subtract error
        results.sort(key=lambda x: float(x.career_position), reverse=True)

        return results


    def _interpret_results(self, avg_position: float, std_position: float) -> str:
        cluster_id = getattr(self, 'current_cluster_id', 'N/A')

        if avg_position > 0.6:
            trend = "Late in their careers"
            confidence = "strong" if std_position < 0.2 else "moderate"
        elif avg_position < 0.4:
            trend = "Early in their careers"
            confidence = "strong" if std_position < 0.2 else "moderate"
        else:
            trend = "throughout their careers (no clear pattern)"
            confidence = "weak"

        interpretation = f"Popular actors tend to work with Cluster {cluster_id} directors {trend}. "
        interpretation += f"This pattern shows {confidence} consistency across actors."

        return interpretation


    def print_detailed_report(self, results: List[ActorCareerAnalysis], top_n: int = 10):
        cluster_id = getattr(self, 'current_cluster_id', 'N/A')
        report = self.generate_report(results, top_n)

        print(f"\nFound {len(results)} popular actors who worked with Cluster {cluster_id} directors\n")

        # Late career actors
        late_career = report['by_phase_mean']['late_career']
        print(f"Late career collaborations (last third): {late_career['count']} actors ({late_career['percentage']:.1f}%)")

        for r in late_career['top_actors'][:5]:
            print(f"\n{r.actor}:")
            print(
                f"  Career: {r.career_start:.0f}-{r.career_end:.0f} ({r.career_length:.0f} years, {r.total_films} films)")
            print(f"  Career position of Cluster {cluster_id} collabs: {r.career_position * 100:.1f}%")
            print(f"  Number of collaborations: {r.num_collabs}")

            # Show up to 3 collaborations
            for c in r.collabs[:3]:
                print(f"    - {c.film} ({c.year:.0f}) with {c.director}")

            if len(r.collabs) > 3:
                print(f"    ... and {len(r.collabs) - 3} more")

        # Early career actors
        early_career = report['by_phase_mean']['early_career']
        print( f"Early career collaborations(first third): {early_career['count']} actors ({early_career['percentage']:.1f}%)")
        print(f"{'=' * 80}")

        for r in early_career['top_actors'][:5]:
            print(f"\n{r.actor}:")
            print(f"  Career: {r.career_start:.0f}-{r.career_end:.0f} ({r.career_length:.0f} years, {r.total_films} films)")
            print(f"  Career position of Cluster {cluster_id} collabs: {r.career_position * 100:.1f}%")
            print(f"  Number of collaborations: {r.num_collabs}")

        # Summary statistics
        summary = report['summary']
        print(f"\nCareer Position Statistics:")
        print(f"  Average: {summary['avg_career_position'] * 100:.1f}%")
        print(f"  Median: {summary['median_career_position'] * 100:.1f}%")
        print(f"  Std Dev: {summary['std_career_position'] * 100:.1f}%")
        print(f"  (0% = career start, 100% = career end)")

        print(f"\nOur interpretation:")
        print(f"{report['interpretation']}")

        # Additional insights
        print(f"\n\nDistribution of mean career phase:")
        for phase in ['early_career', 'middle_career', 'late_career']:
            phase_data = report['by_phase_mean'][phase]
            print(f"{phase.replace('_', ' ').title()}: {phase_data['count']} actors ({phase_data['percentage']:.1f}%)")

        patterns = {
            'all_phases': 'Throughout career (all 3 phases)',
            'never': 'Never collaborated',
            'early_only': 'Early career only',
            'middle_only': 'Middle career only',
            'late_only': 'Late career only',
            'early_middle': 'Early to middle career',
            'middle_late': 'Middle to late career',
            'early_late': 'Early and late (but not middle)'
        }

        print(f"\n\nDistribution of the career styles??:")
        for key, value in report['by_phase'].items():
            print(f"{patterns[key]}: \n{value['count']} actors ({value['percentage']:.1f}%)")





    def generate_report(self, results: List[ActorCareerAnalysis], top_n: int = 10) -> Dict:
        if not results:
            return {"error": "No results to analyze"}

        # Categorize by career phase
        late_career = [r for r in results if r.career_position > self.LATE_CAREER]
        early_career = [r for r in results if r.career_position < self.EARLY_CAREER]
        middle_career = [r for r in results if self.EARLY_CAREER <= r.career_position <= self.LATE_CAREER]

        # Calculate statistics
        all_positions = [r.career_position for r in results]
        avg_position = np.mean(all_positions)
        median_position = np.median(all_positions)
        std_position = np.std(all_positions)

        # Calculate collaboration statistics
        total_collabs = sum(r.num_collabs for r in results)
        avg_collabs_per_actor = np.mean([r.num_collabs for r in results])

        all_phase = 0
        never = 0
        early_only = 0
        middle_only = 0
        late_only = 0
        early_middle = 0
        middle_late = 0
        early_late = 0

        for r in results:
            if r.actor_phase_summary == "all_phases":
                all_phase += 1
            elif r.actor_phase_summary == "never":
                never += 1
            elif r.actor_phase_summary == "early_only":
                early_only += 1
            elif r.actor_phase_summary == "middle_only":
                middle_only += 1
            elif r.actor_phase_summary == "late_only":
                late_only += 1
            elif r.actor_phase_summary == "early_middle":
                early_middle += 1
            elif r.actor_phase_summary == "early_late":
                early_late += 1
            elif r.actor_phase_summary == "middle_late":
                middle_late += 1



        report = {
            "summary": {
                "total_actors_analyzed": len(results),
                "avg_career_position": avg_position,
                "median_career_position": median_position,
                "std_career_position": std_position,
                "total_collaborations": total_collabs,
                "avg_collabs_per_actor": avg_collabs_per_actor
            },
            "by_phase_mean": {
                "late_career": {
                    "count": len(late_career),
                    "percentage": len(late_career) / len(results) * 100,
                    "top_actors": late_career[:top_n]
                },
                "middle_career": {
                    "count": len(middle_career),
                    "percentage": len(middle_career) / len(results) * 100,
                    "top_actors": middle_career[:top_n]
                },
                "early_career": {
                    "count": len(early_career),
                    "percentage": len(early_career) / len(results) * 100,
                    "top_actors": early_career[:top_n]
                }
            },

            "by_phase": {
                "all_phases": {
                    "count": all_phase,
                    "percentage": all_phase / len(results) * 100,
                },
                "never": {
                    "count": never,
                    "percentage":never / len(results) * 100,
                },
                "early_only": {
                    "count": early_only,
                    "percentage": early_only / len(results) * 100,
                },
                "middle_only": {
                    "count": middle_only,
                    "percentage": middle_only / len(results) * 100,
                },
                "late_only": {
                    "count": late_only,
                    "percentage": late_only / len(results) * 100,
                },
                "early_middle": {
                    "count": early_middle,
                    "percentage": early_middle / len(results) * 100,
                },
                "middle_late": {
                    "count": middle_late,
                    "percentage": middle_late / len(results) * 100,
                },
                "early_late": {
                    "count": early_late,
                    "percentage":early_late / len(results) * 100,
                },
            }
            ,
            "interpretation": self._interpret_results(avg_position, std_position)
        }

        return report


