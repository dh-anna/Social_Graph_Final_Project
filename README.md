# Social_Graph_Final_Project

## Dataset Setup
To download the dataset, run the `Dataset_downloader.ipynb` notebook. The dataset will be downloaded into the Dataset folder.

## Report
The report can be found as a pdf in the root folder, named...

## 3D visualization of the actor-director graph
https://dh-anna.github.io/Social_Graph_Final_Project/

## Brief overview

This project examines whether working with established directors affects an actor's career trajectory. We analyze this question through two main network analysis approaches:

### Actor-Director Graph
This network connects actors to the directors they have worked with, allowing us to analyze collaboration patterns and identify how working with prominent directors correlates with actor career success.

### Director-Director Graph
This network connects directors based on how actors moved between them.

## Key Findings

### Actor-Director Graph Analysis
This work is in the Artist_Director_Graph folder. The Artist_Director_Graph.ipynb is the main jupyter notebook, the helper functions are located in that folder as well.
- Community detection revealed distinct clusters of actors and directors with specialized characteristics (genres, production companies, time periods)
- Nodes with highest degree centrality are established directors
- Different clusters does not show specialization in specific genres, production companies, and film eras
- Sentiment analysis did not show any correlation with movie sentiment and actor popularity

### Director-Director Graph Analysis
This work is in the Director_Director_Graph folder. The Director_Director_Graph.ipynb is the main jupyter notebook, the helper functions are located in director_graph_functions.py.
- We observed the career movement of actors between directors
- **Network Distance Effect**: Strong negative correlation (r = -0.34, p < 0.001) between network distance to top 10 directors and director popularity, indicating that proximity to established directors matters for career success
- **Memory-Based Career Model**: Actors' past collaborations strongly predict future opportunities:
  - High-popularity actors are significantly more likely to continue working with high-popularity directors, but low popularity actors can work with high popularity directors too, we could not identify a lock-in-effect

### Conclusion:
- In the artist–director graph, we identified 30 communities, but we found no correlation between the community structure and any observable features such as genre, time range, production company, or the sentiment of the movies. This may be due to unobserved factors, such as the packaging phenomena from talent agencies, for which no data were available.

- We also found no correlation between actors’ popularity and the sentiment of the movies in which they appeared.

- Furthermore, an actor’s first three movies do not reliably predict the popularity of their last three movies.

- Finally, we demonstrated that our memory term, based on the popularity of directors with whom an actor has previously worked, is a strong indicator of whether an actor will transition to a popular director. Actors with high memory terms are nearly 3.55 times more likely than those with low memory terms to collaborate with high-popularity directors.

