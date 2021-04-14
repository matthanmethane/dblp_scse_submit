*** "$ pip install -r requirements.txt" before opening the GUI ***

Guide for GUI:

1. Click Load File Button \
	Upload your Faculty.xlsx \
	Upload your Top.xlsx \
2. Click Initialize files \
	This will create all the txt/xml needed for the program to run \
3. Click Next Button \
	Make sure that files are initialized \ 



\
\
\




1. get_coworker_dict()
	This will create "weighted_collab.txt" which contains edges of the graph
2. ret_graph():
	This will read the "weighted_collab.txt" and return the NetworkX Graph
3. centrality_to_excel(G)
	This will create an Excel file of all centralities 
4. centrality_to_dataframe(G,cen_type)
	This will return a Pandas DataFrame for the input Centrality
	['all,'degree','eigenvector','betweenness']
5. no_top_venue_dataframe
	This will return a Pandas DataFrame of Number in top venues
6. centrality_top_venue_dataframe(G)
	This will return a Pandas DataFrame of Centralities and Number in top venues 
7. centrality_top_venue_scatter(G, cen_type)
	This will plot a scatter plot between Centrality and Number in top venue
# dblp_scse_submit
