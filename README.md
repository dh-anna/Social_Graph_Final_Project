# Social_Graph_Final_Project

## Dataset Setup
To download the dataset, run the `Dataset_downloader.ipynb` notebook. The dataset will be downloaded into the Dataset folder.

## Overview

This project examines whether working with established directors affects an actor's career trajectory. We analyze this question through two main network analysis approaches:

### Actor-Director Graph
This network connects actors to the directors they have worked with, allowing us to analyze collaboration patterns and identify how working with prominent directors correlates with actor career success.

### Director-Director Graph
This network connects directors based on how actors moved between them.

## Key Findings

### Actor-Director Graph Analysis
This work is in the Artist_Director_Graph folder. The Artist_Director_Graph.ipynb is the main jupyter notebook, the helper functions are located in actor_director_analysis.py and actor_director_functions.py.
- Community detection revealed distinct clusters of actors and directors with specialized characteristics (genres, production companies, time periods)
- Nodes with highest degree centrality are established directors
- Actors who worked with popular directors in early career showed higher career success rates
- Different clusters does not show specialization in specific genres, production companies, and film eras

### Director-Director Graph Analysis
This work is in the Director_Director_Graph folder. The Director_Director_Graph.ipynb is the main jupyter notebook, the helper functions are located in director_graph_functions.py.
- We observed the career movement of actors between directors
- **Network Distance Effect**: Strong negative correlation (r = -0.34, p < 0.001) between network distance to top 10 directors and director popularity, indicating that proximity to established directors matters for career success
- **Memory-Based Career Model**: Actors' past collaborations strongly predict future opportunities:
  - High-reputation actors are significantly more likely to continue working with high-prestige directors
  - Low-reputation actors tend to remain working with lower-prestige directors
  - This suggests a "reputation lock-in" effect where early career choices have lasting impact


