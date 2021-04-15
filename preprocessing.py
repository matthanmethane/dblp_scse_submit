import pandas as pd
from faculty import Faculty, get_xml_link, load_faculty_xml
from bs4 import BeautifulSoup
from math import log
import lxml
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import collections
import time
import contextlib
import itertools
import operator
import xml.etree.ElementTree as ET


#Network Property
############################################################################################################
def get_coworker_graph(nodes, year = 2021, mode = "all", weighted = True):
    edges = []
    for pid_string in nodes:
        file = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding = 'utf-8')
        content = BeautifulSoup(file,"lxml")
        file.close()

        contents_r = content.findAll("r")
        for content_r in contents_r:
            content_year = int(content_r.find("year").text)
            #filter out content after year
            if (content_year <= year):
                coauthors = content_r.findAll("author")
                for coauthor in coauthors:
                    if((coauthor["pid"] in nodes) and (pid_string<coauthor["pid"])):
                        edge = (pid_string,coauthor["pid"])
                        edges.append(edge)

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    
    if weighted:
        #give weight to edges
        for i in edges:
            G[i[0]][i[1]]['weight']=edges.count(i)

    #use mode="connected" to filter out unconnected nodes
    if(mode=="connected"):
        isolates = list(nx.isolates(G))
        G.remove_nodes_from(isolates)
    #use mode="giant" to choose giant component only
    if(mode=="giant"):
        giant = G.subgraph(max(nx.connected_components(G), key=len))
        return giant
    return(G)
def get_properties(G):
    number_of_nodes=G.number_of_nodes()
    number_of_edges=G.number_of_edges()
    average_degree=number_of_edges/number_of_nodes
    average_clustering=nx.average_clustering(G)
    #print("number of nodes: n =", number_of_nodes)
    #print("number of edges: ∑k =", number_of_edges)
    #print("average degree: <k> =", average_degree)
    #print("average clustering coefficient: <C> =", average_clustering)
    #print("diameter: d = ", end="")
    try: 
        diameter=nx.diameter(G)
        #print(diameter)
    except (nx.exception.NetworkXError):
        diameter = None
        #print("The graph is not connected")
    #print("average distance: <d> = ", end="")
    try: 
        average_distance = nx.average_shortest_path_length(G)
        #print(average_distance)
    except (nx.exception.NetworkXError):
        average_distance = None
        #print("The graph is not connected")

        
    degree_centrality = nx.degree_centrality(G)
    highest_degree_centrality_pid = max(degree_centrality, key=degree_centrality.get)
    highest_degree_centrality_value = degree_centrality.get(highest_degree_centrality_pid)
    #print("highest degree centrality:", highest_degree_centrality_pid, highest_degree_centrality_value)
    
    #default using max_iter = 100, tolerance=10^-6
    #eigen_centrality = nx.eigenvector_centrality_numpy(G)
    eigen_centrality = nx.eigenvector_centrality(G)
    highest_eigen_centrality_pid = max(eigen_centrality, key=eigen_centrality.get)
    highest_eigen_centrality_value = eigen_centrality.get(highest_eigen_centrality_pid)
    #print("highest eigen centrality:", highest_eigen_centrality_pid, highest_eigen_centrality_value)
    
    closeness_centrality = nx.closeness_centrality(G)
    highest_closeness_centrality_pid = max(closeness_centrality, key=closeness_centrality.get)
    highest_closeness_centrality_value = closeness_centrality.get(highest_closeness_centrality_pid)
    #print("highest closeness centrality:", highest_closeness_centrality_pid, highest_closeness_centrality_value)
    
    betweenness_centrality = nx.betweenness_centrality(G)
    highest_betweenness_centrality_pid = max(betweenness_centrality, key=betweenness_centrality.get)
    highest_betweenness_centrality_value = betweenness_centrality.get(highest_betweenness_centrality_pid)
    #print("highest betweenness centrality:", highest_betweenness_centrality_pid, highest_betweenness_centrality_value)
    
    #return a list of network properties
    result = []
    for i in (number_of_nodes,number_of_edges,average_degree,average_clustering,diameter,average_distance,highest_degree_centrality_pid,highest_degree_centrality_value,highest_eigen_centrality_pid,highest_eigen_centrality_value,highest_closeness_centrality_pid,highest_closeness_centrality_value,highest_betweenness_centrality_pid,highest_betweenness_centrality_value):
        result.append(i)
    return result

def ret_graph_network_year(yr_input=2021, mode_input ="connected",weighted_bool = False):
    nodes = ret_nodes()
    G = get_coworker_graph(nodes, year = yr_input, mode = mode_input, weighted = weighted_bool)
    return G
    #get_properties(G)

#return properties dataframe
def get_properties_yearly(nodes,year=2000, mode_input="all"):
    with contextlib.redirect_stdout(None):
        result_dict = {"Year":[],"nodes":[],"edges":[],"average_degree":[],"average_clustering":[],"diameter":[],"average_distance":[],"highest_degree_centrality_pid":[],"highest_degree_centrality_value":[],"highest_eigen_centrality_pid":[],"highest_eigen_centrality_value":[],"highest_closeness_centrality_pid":[],"highest_closeness_centrality_value":[],"highest_betweenness_centrality_pid":[],"highest_betweenness_centrality_value":[]}
        dict_keys=["Year","nodes","edges","average_degree","average_clustering","diameter","average_distance","highest_degree_centrality_pid","highest_degree_centrality_value","highest_eigen_centrality_pid","highest_eigen_centrality_value","highest_closeness_centrality_pid","highest_closeness_centrality_value","highest_betweenness_centrality_pid","highest_betweenness_centrality_value"]
        for i in range(int(time.strftime("%Y")),year-1,-1):
            G = get_coworker_graph(nodes, year = i, mode = mode_input)
            result = get_properties(G)
            result.insert(0,i) 
            for i in range(len(result)):
                result_dict[dict_keys[i]].append(result[i])

    result_df = pd.DataFrame.from_dict(result_dict)
    return result_df

def yearly_diff(df,N=1):
    shift_range = N+1
    merging_keys = ['Year']
    lag_cols = ['nodes','edges','average_degree','average_clustering']
    for shift in range(1,shift_range):
        df_shift = df[merging_keys + lag_cols].copy()

        # E.g. Year of 2000 becomes 2001, for shift = 1.
        # So when this is merged with Year of 2001 in df, this will represent lag of 2001.
        df_shift['Year'] = df_shift['Year'] + shift
        foo = lambda x: '{}_lag_{}'.format(x, shift) if x in lag_cols else x
        df_shift = df_shift.rename(columns=foo)
        df = pd.merge(df, df_shift, on=merging_keys, how='left') #.fillna(0)

        bar = lambda x: '{}_diff_{}'.format(x, shift) if x in lag_cols else x
        for i in lag_cols:
            df[bar(i)]=df[i]-df[foo(i)]
            df.drop(foo(i), axis=1, inplace=True)
        df = df.astype({bar('nodes'):pd.Int64Dtype()})
        df = df.astype({bar('edges'):pd.Int64Dtype()})
    del df_shift
    return df 

def ret_nodes():
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    file.close()
    return pid_list_rstrip




#Collaboration
############################################################################
def find_name_with_pid(pid,faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        faculty = Faculty(df_line["Faculty"],pid_list_rstrip[idx],df_line["Position"],df_line["Gender"],df_line["Management"],df_line["Area"])
        faculty_list.append(faculty)
    for faculty in faculty_list:
        if(faculty.pid == pid):
            print(faculty.name)
            return faculty.name

# Find "Position" (rank) using pid
def find_pos_with_pid(pid,faculty_path):
    pos_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        pos = Faculty(df_line["Position"],pid_list_rstrip[idx],df_line["Faculty"],df_line["Gender"],df_line["Management"],df_line["Area"])
        pos_list.append(pos)
    for pos in pos_list:
        #print(pos)
        if(pos.pid == pid):
            return pos.name

# Find "Management" (or not) using pid
def find_man_with_pid(pid,faculty_path):
    man_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        man = Faculty(df_line["Management"],pid_list_rstrip[idx],df_line["Faculty"],df_line["Position"],df_line["Gender"],df_line["Area"])
        man_list.append(man)
    for man in man_list:
        # print(man)
        if(man.pid == pid):
            return man.name

# Find "Area" using pid
def find_area_with_pid(pid,faculty_path):
    area_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        area = Faculty(df_line["Area"],pid_list_rstrip[idx],df_line["Faculty"],df_line["Position"],df_line["Gender"],df_line["Management"])
        area_list.append(area)
    for area in area_list:
        # print(area)
        if(area.pid == pid):
            return area.name
            
def get_coworker_dict(faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        faculty = Faculty(df_line["Faculty"],pid_list_rstrip[idx],df_line["Position"],df_line["Gender"],df_line["Management"],df_line["Area"])
        faculty_list.append(faculty)
    

    coauthor_dict = {}
    pid_strings = [faculty.pid for faculty in faculty_list]
    for pid_string in pid_strings:
        file = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding='utf-8') 
        content = BeautifulSoup(file,"lxml")
        file.close()
        coauthor_pane = content.find("coauthors")
        
        coauthors = coauthor_pane.findAll("na")
        coauthor_pid_list = []
        for coauthor in coauthors:
            try:
                if coauthor["pid"] in pid_list_rstrip:
                    coauthor_pid_list.append(coauthor["pid"])
            except:
                continue
        coauthor_dict[pid_string] = coauthor_pid_list

    return(coauthor_dict)

def degree_histogram(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    for key in degreeCount:
        degreeCount[key] = degreeCount[key]/(G.number_of_edges())
    deg, cnt = zip(*degreeCount.items())

    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color="b")

    plt.title("Degree Histogram")
    plt.ylabel("Probability")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)

    # draw graph in inset
    plt.axes([0.4, 0.4, 0.5, 0.5])
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(G)
    plt.axis("off")
    nx.draw_networkx_nodes(G, pos, node_size=20)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()

def degree_histogram_loglog(G):
    degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    degreeCountLog = {}
    for key in degreeCount:
        try:
            log_pk = log(key)
            log_k = log(degreeCount[key]/(G.number_of_edges()))
            degreeCountLog[log_pk] = log_k
        except:
            continue
    deg, cnt = zip(*degreeCountLog.items())
    plt.scatter(deg,cnt)
    plt.title("log-log Degree Histogram")
    plt.ylabel("log Probability")
    plt.xlabel("log Degree")
    plt.show()

# node_dict = get_coworker_dict()

# G = nx.Graph(node_dict)


def init_collab(faculty_path):
    node_dict = get_coworker_dict(faculty_path)
    G = nx.Graph(node_dict)
    nx.write_edgelist(G, "edge_list.txt", delimiter=' ', data=False) # Generate edge_list.txt
    

#Collaborative Property
#######################################################
def init_collab_network():
    for i in range(int(time.strftime("%Y")),1999,-1):
        nodes = ret_nodes()
        G = get_coworker_graph(nodes, year = i, mode = "connected")
        path = r"edge_lists/"
        nx.write_edgelist(G, path+str(i) + "_edge_list.txt", delimiter=' ', data=False) # Generate edge_list.txt (yearly)


def ret_collab_network(collab_type, pid,faculty_path):
    fig_count = 0
    pid_real = pid.replace('_','/')
    print(pid_real)
    # Set True to select collaborative property to plot
    num_collab = False
    rank_collab = False
    man_collab = False
    area_collab = False
    if(collab_type=="num_collab"):
        num_collab = True
    elif(collab_type=="rank_collab"):
        rank_collab = True
    elif(collab_type=="man_collab"):
        man_collab = True
    elif(collab_type=="area_collab"):
        area_collab = True

    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    ax = axes.flatten()

    for i in range(int(time.strftime("%Y")),1999,-1):
        file = open(f'edge_lists/{str(i)+"_edge_list"}.txt','r')
        G = str(i)
        # print(G)
        G = nx.Graph()
        for line in file:
            a, b = line.split()
            # change faculty member name to track 
            if (a in pid_real or b in pid_real):
                # print(find_name_with_pid(a))

                if num_collab:
                    n1 = a
                    n2 = b

                    G.add_node(n1)
                    G.add_node(n2)
                
                if rank_collab:
                    n1 = find_pos_with_pid(a,faculty_path)
                    n2 = find_pos_with_pid(b,faculty_path)

                    G.add_node(n1)
                    G.add_node(n2)

                if man_collab:
                    n1 = find_man_with_pid(a,faculty_path)
                    n2 = find_man_with_pid(b,faculty_path)
                    
                    G.add_node(n1)
                    G.add_node(n2)
                
                if area_collab:
                    n1 = find_area_with_pid(a,faculty_path)
                    n2 = find_area_with_pid(b,faculty_path)
                    
                    G.add_node(n1)
                    G.add_node(n2)

                if G.has_edge(n1, n2):
                    # Increase weight by 1
                    G[n1][n2]['weight'] += 1
                else:
                    # new edge. add with weight = 1
                    G.add_edge(n1, n2, weight=1)

        # plt.figure(fig_count)
        pos = nx.spring_layout(G, seed=7)
        labels = nx.get_edge_attributes(G,'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        nx.draw(G, pos, with_labels=True, ax=ax[fig_count])
        ax[fig_count].set_axis_off()
        plt.tight_layout()

        fig_count = fig_count + 1

    fig.delaxes(axes[4, 2])
    fig.delaxes(axes[4, 3])
    fig.delaxes(axes[4, 4])
    # plt.tight_layout()
    plt.show()

#Excellency and Centrality
########################################################
#Function to return list of all Faculty class
def get_faculty_list(faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        faculty = Faculty(df_line["Faculty"],pid_list_rstrip[idx],df_line["Position"],df_line["Gender"],df_line["Management"],df_line["Area"])
        faculty_list.append(faculty)
    return faculty_list


#Function to get pid: coworker Dictonary and create file 'Weighted_collab.txt'
def get_coworker_dict_cent(faculty_path):
    faculty_list = get_faculty_list(faculty_path)
    coauthor_dict = {}
    pid_strings = [faculty.pid for faculty in faculty_list]
    with open("weighted_collab.txt","w", encoding='utf-8') as f:
        for pid_string in pid_strings:
            file = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding='utf-8') 
            content = BeautifulSoup(file,"lxml")
            file.close()
            coauthor_pane = content.find("coauthors")
            coauthors = coauthor_pane.findAll("na")
            coauthor_pid_list = []
            for coauthor in coauthors:
                try:
                    if coauthor["pid"] in pid_strings:
                        collab_pid = coauthor["pid"]
                        author_pane = content.findAll("author",{"pid":coauthor["pid"]})
                        no_collab = len(author_pane)
                        f.write(f"{pid_string}\t{collab_pid}\t{no_collab}\n")
                        print(f"Writing... {pid_string} {collab_pid} {no_collab}")
                except Exception as e:
                    print(e)
                    continue
            coauthor_dict[pid_string] = coauthor_pid_list
    return(coauthor_dict)

def ret_graph_cent():  
    G = nx.Graph()
    try:
        with open("weighted_collab.txt","r", encoding='utf-8') as f:
            G = nx.read_weighted_edgelist(f)
    except Exception as e:
        print(e)
        return G
    return G


#Function to get pid: Area Dictionary
def get_area_dict(faculty_path):
    faculty_list = get_faculty_list(faculty_path)
    
    area_dict = {}
    for faculty in faculty_list:
        area_dict[faculty.pid] = faculty.area

    return(area_dict)

#Function to create a heatmap Graph
def draw_heatmap(G, measures, measure_name, node_bool = True):
    pos = nx.spring_layout(G)
    nodes = nx.draw_networkx_nodes(G, pos, node_size=250,
                                 cmap = plt.cm.plasma,
                                 node_color = list(measures.values()),
                                 nodelist=list(measures.keys()))
    nodes.set_norm(mcolors.SymLogNorm(linthresh=0.01, linscale=1))
    edges = nx.draw_networkx_labels(G, pos)
    plt.title(measure_name)
    plt.colorbar(nodes)
    plt.axis("off")
    plt.show()


#Function to create an Excel file of all centralities
def centrality_to_excel(G):
    #Get values of Centrality
    deg_cen = nx.degree_centrality(G)
    eigen_cen = nx.eigenvector_centrality(G)
    betweeness_cen = nx.betweenness_centrality(G)
    #Extract list to append
    eigen_cen_vals = [eigen_cen[key] for key in eigen_cen]
    betweeness_cen_vals = [betweeness_cen[key] for key in betweeness_cen]
    #Create a Centrality Table
    df_cen = pd.DataFrame.from_dict(deg_cen.items())
    df_cen.columns = ['ID','Degree Centrality']
    df_cen['Eigenvector Centrality'] = eigen_cen_vals
    df_cen['Betweenness Centrality'] = betweeness_cen_vals
    print(df_cen)
    df_cen.to_excel("Centrality.xlsx")  

#Function to return a DataFrame of each Centrality
def centrality_to_dataframe(G, cen_type="all"):
    if(cen_type == "all"):
        #Get values of Centrality
        deg_cen = nx.degree_centrality(G)
        eigen_cen = nx.eigenvector_centrality(G)
        betweeness_cen = nx.betweenness_centrality(G)
        #Extract list to append
        eigen_cen_vals = [eigen_cen[key] for key in eigen_cen]
        betweeness_cen_vals = [betweeness_cen[key] for key in betweeness_cen]
        #Create a Centrality Table
        df_cen = pd.DataFrame.from_dict(deg_cen, orient='index')
        df_cen.columns = ['Degree Centrality']
        df_cen['Eigenvector Centrality'] = eigen_cen_vals
        df_cen['Betweenness Centrality'] = betweeness_cen_vals
        sort_df_cen = df_cen.sort_values(by = ["Degree Centrality"], ascending=[False])
        return sort_df_cen
    elif(cen_type == "degree"):
        #Get values of Centrality
        deg_cen = nx.degree_centrality(G)
        #Create sorted Degree Centrality Table
        df_deg_cen = pd.DataFrame.from_dict(deg_cen, orient='index')
        df_deg_cen.columns = ['Degree Centrality']
        sort_df_deg_cen = df_deg_cen.sort_values(by = ['Degree Centrality'],ascending=[False])
        return(sort_df_deg_cen)
    elif(cen_type == "eigenvector"):
        #Get values of Centrality
        eigen_cen = nx.eigenvector_centrality(G)
        #Create sorted Eigenvector Centrality Table
        df_eigen_cen = pd.DataFrame.from_dict(eigen_cen, orient='index')
        df_eigen_cen.columns = ['Eigenvector Centrality']
        sort_df_eigen_cen = df_eigen_cen.sort_values(by = ['Eigenvector Centrality'],ascending=[False])
        return(sort_df_eigen_cen)
    elif(cen_type == "betweenness"):
        #Get values of Centrality
        betweeness_cen = nx.betweenness_centrality(G)
        #Create sorted Betweeness Centrality Table
        df_betweeness_cen = pd.DataFrame.from_dict(betweeness_cen, orient='index')
        df_betweeness_cen.columns = ['Betweenness Centrality']
        sort_df_betweeness_cen = df_betweeness_cen.sort_values(by = ['Betweenness Centrality'],ascending=[False])
        return(sort_df_betweeness_cen)
    else:
        return None

#Function to get DataFrame of Professor and number of Publications in the Top Venues
def no_top_venue_dataframe(faculty_path):
    VENUE_DICT = {
        "Data Management" : "sigmod",
        "Data Mining" : "kdd",
        "Information Retrieval" : "sigir",
        "Computer Vision" : "cvpr",
        "AI/ML" : "nips",
        "Computer Networks" : "sigcomm",
        "Cyber Security" : "ccs",
        "Software Engg" : "icse",
        "Computer Architecture" : "isca",
        "HCI" : "chi",
        "Distributed Systems" : "podc",
        "Computer Graphics" : "siggraph",
        "Bioinformatics" : "recomb",
        "Multimedia" : "mm"
    }
    VENUE_LIST = [VENUE_DICT[key] for key in VENUE_DICT]
    CONFERENCE_NAME = {
        "sigmod" : "SIGMOD Conference",
        "kdd" : "KDD",
        "sigir" : "SIGIR",
        "cvpr" : "CVPR",
        "nips" : "NeurIPS",
        "sigcomm" : "SIGCOMM",
        "ccs" : "CCS",
        "icse" : "ICSE",
        "isca" : "ISCA",
        "chi" : "CHI",
        "podc" : "PODC",
        "siggraph" : "SIGGRAPH" , #contain SIGGRAPH without @
        "recomb" : "RECOMB",
        "mm" : "ACM Multimedia",
    }

    faculty_list = get_faculty_list(faculty_path)
    area_list = get_area_dict(faculty_path)

    #TODO: The venues should match with the professor's respective subject
    faculty_venue_dict = {}
    pid_strings = [faculty.pid for faculty in faculty_list]
    for pid_string in pid_strings:
        file = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding='utf-8') 
        content = BeautifulSoup(file,"lxml")
        file.close()
        publications_pane = content.find_all("inproceedings")
        venue_list = []
        for publications in publications_pane:
            #Check if the publication is a workshop or not
            publication_booktitle = publications.find("booktitle").get_text()
            #Add to list if it is a conference, Must exclude workshops
            publication = publications["key"]
            publication_split = publication.split('/')
            publication_venue = publication_split[1]    
            if(publication_venue == VENUE_DICT[area_list[pid_string]]):
                if(publication_venue == "siggraph" and publication_booktitle == CONFERENCE_NAME[publication_venue] and not("\@" in publication_booktitle)):
                    venue_list.append(publication_venue)
                elif(publication_booktitle == CONFERENCE_NAME[publication_venue]):
                    venue_list.append(publication_venue)
        faculty_venue_dict[pid_string] = venue_list

    faculty_venue_no_dict = {}
    for key in faculty_venue_dict:
        faculty_venue_no_dict[key] = len(faculty_venue_dict[key])

    df = pd.DataFrame.from_dict(faculty_venue_no_dict, orient='index')
    df.columns = ["No.Publication in Top Venue"]
    sorted_df = df.sort_values(by = ['No.Publication in Top Venue'],ascending=[False])
    return sorted_df

#Function to return dataframe of all centralities and number of pulications in top venue
def centrality_top_venue_dataframe(G,faculty_path):
    df_centrality = centrality_to_dataframe(G)
    df_top_venue = no_top_venue_dataframe(faculty_path)
    df_centrality["No.Publication in Top Venue"] = df_top_venue["No.Publication in Top Venue"]
    df_centrality = df_centrality.rename(index = lambda x: find_name_with_pid(x,faculty_path))
    return(df_centrality)
#Function to show graph between centrality and number of pulications in top venue
def centrality_top_venue_scatter(G, cen_type,faculty_path):
    df = centrality_top_venue_dataframe(G,faculty_path)
    plt.scatter(df[f'{cen_type} Centrality'], df['No.Publication in Top Venue'])
    plt.title(f"{cen_type} Centrality VS No. Publication")
    plt.xlabel(f"{cen_type} Centrality")
    plt.ylabel("Number of Publication in Top Venue ")
    plt.show()
    

########################################################

def find_all_coauthors(faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        faculty = Faculty(df_line["Faculty"],pid_list_rstrip[idx],df_line["Position"],df_line["Gender"],df_line["Management"],df_line["Area"])
        faculty_list.append(faculty)
        
    coauthor_dict = {}
    pid_strings = [faculty.pid for faculty in faculty_list]
    for pid_string in pid_strings:
        xmlDoc = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding='utf-8')
        xmlDocData = xmlDoc.read()
        xmlDocTree = ET.XML(xmlDocData)
                      
        coauthor_pid_list = []
                      
        for CA in xmlDocTree.iter('author'):
            
            if CA.attrib['pid'] not in pid_list_rstrip:     #not in NTU
                coauthor_pid_list.append(CA.attrib['pid'])         #repetition allowed for the question
        xmlDoc.close()             
        coauthor_dict[pid_string] = coauthor_pid_list
        
                      
    return(coauthor_dict)





def name_with_pid(faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    for idx, df_line in df.iterrows():
        faculty = Faculty(df_line["Faculty"],pid_list_rstrip[idx],df_line["Position"],df_line["Gender"],df_line["Management"],df_line["Area"])
        faculty_list.append(faculty)
        
    coauthor_dict = {}
    pid_strings = [faculty.pid for faculty in faculty_list]
    for pid_string in pid_strings:
        xmlDoc = open(f'faculty_xml/{pid_string.replace("/","_")}.xml','r',encoding='utf-8')
        xmlDocData = xmlDoc.read()
        xmlDocTree = ET.XML(xmlDocData)
                      
        coauthor_name_with_pid = []
        temp = []            
        for CA in xmlDocTree.iter('author'):
            
            if CA.attrib['pid'] not in pid_list_rstrip:     #not in NTU
                temp = [CA.attrib['pid'],CA.text]
                coauthor_name_with_pid.append(temp)   
                temp=[]                                 #repetition allowed for the question
        xmlDoc.close()
        coauthor_dict[pid_string] =  coauthor_name_with_pid
    ll=[]
    for g in coauthor_dict:
        for i in coauthor_dict[g]:
            if i not in ll:
                ll.append(i)
    df = pd.DataFrame(ll)
    return ll
        
                      


def keys_return(faculty_path):
    faculty_list = []
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt","r")
    pid_list = file.readlines()
    pid_list_rstrip = [pid.replace("_",'/').rstrip() for pid in pid_list]
    return pid_list_rstrip


########################################################
def get_dict_of_edges(faculty_path):
    
    dictForAllCoAuthors = find_all_coauthors(faculty_path)    
    listForEligible = []
    dictOfEdges = {}
    x=0
    keylist = keys_return(faculty_path)
    for i in range(len(keylist)):
        tempList = []
        for j in range(len(dictForAllCoAuthors[keylist[i]])):
            tempPid = (dictForAllCoAuthors[keylist[i]][j])
            count = 0
            for h in range(len(dictForAllCoAuthors[keylist[i]])):
                if tempPid == dictForAllCoAuthors[keylist[i]][h]:
                    count = count + 1
            if count > 4 and tempPid not in listForEligible:   # count > 4 in order to account for the repeat.
                listForEligible.append(tempPid)
                tempList.append(tempPid)
        dictOfEdges[keylist[i]] = tempList            
    
    return dictOfEdges


def graph_Qn7(dictOfEdges):
    G = nx.Graph(dictOfEdges)
    return G

def draw_graph(G):
        d = dict(G.degree)
        color_map = []
        for node in G:
            if len(G.edges(node))<1:
                color_map.append('blue')
            elif len(G.edges(node)) > 1 and len(G.edges(node))< 10:
                color_map.append('green')
            elif len(G.edges(node)) > 10 and len(G.edges(node)) < 20:
                color_map.append('yellow')
            else: 
                color_map.append('red')      
        nx.draw(G, node_size = 10,node_color=color_map, with_labels=False)
        plt.show()
def get_graph_sorted_degree(G):
    result=[]
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')
    sorted_degree = sorted(degree_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("Top 20 nodes by degree:")
    for d in sorted_degree[:20]:
        print(d)
        result.append(d)
    return result

def get_graph_degree_cen(G):
    result=[]
    degree_cen_dict = nx.degree_centrality(G) # run degree centrality
    nx.set_node_attributes(G, degree_cen_dict , 'degree')
    
    sorted_degreeC = sorted(degree_cen_dict.items(), key=operator.itemgetter(1), reverse=True)
    df0= pd.DataFrame(sorted_degreeC)
    df0.columns = ['Nodes','Degree Centrality']
    print("Top 20 nodes by degree centrality:")
    for b in sorted_degreeC[:20]:
        print(b)
        result.append(b)
    return result

def get_graph_betweeness(G):
    result=[]
    betweenness_dict = nx.betweenness_centrality(G) # run betweenness centrality
    nx.set_node_attributes(G, betweenness_dict, 'betweenness')
    sorted_betweenness = sorted(betweenness_dict.items(), key=operator.itemgetter(1), reverse=True)
    print("Top 20 nodes by betweenness centrality:")
    df1= pd.DataFrame(sorted_betweenness)
    df1.columns = ['Nodes','Betweeness Centrality']
    for b in sorted_betweenness[:20]:
        print(b)
        result.append(b)
    return result


def get_graph_eigen(G):
    result=[]
    eigenvector_dict = nx.eigenvector_centrality(G) # run eigenvector centrality
    nx.set_node_attributes(G, eigenvector_dict, 'eigenvector')
    sorted_eigenvector = sorted(eigenvector_dict.items(), key=operator.itemgetter(1), reverse=True)
    df2=pd.DataFrame(sorted_eigenvector)
    df2.columns = ['Nodes','Eigenvector Centrality']
    print("Top 20 nodes by eigenvector centrality:")
    for b in sorted_eigenvector[:20]:
        print(b)
        result.append(b)
    return result




def get_dict_of_field(faculty_path):
    data = pd.read_excel(faculty_path)
    df = pd.DataFrame(data, columns=["Faculty","Position","Gender","Management","Area"])
    file = open("pid.txt",'r')
    pidlist = []
    for lines in file.readlines():
        pidlist.append(lines.strip('\n'))
    df['pid'] = pidlist


    df1 = df[['Area', "pid"]]


    g= list(df1['pid'])
    words = [w.replace('_', '/') for w in g]

    df1.drop(columns = 'pid')
    df1['pida']= words

    cyberSec = []
    ai_ml = []
    comVision = []
    comArc = []
    multiM=[]
    comNet = []
    distSys = []
    bioM=[]
    ir = []
    listOfField=[]
    for area in df1['Area']:
        if area not in listOfField:
            listOfField.append(area)

    dictOfField = {}
    for i in listOfField:
        listm=[]
        for ind in df1.index:
            if(str(df1['Area'][ind]) == str(i)):
                listm.append(df1['pida'][ind])
        dictOfField[i] =listm
        
    return dictOfField  

def get_total_number(dictOfField,dictOfEdges):
    noDict ={}
    for keys in dictOfField:
        noDict[keys]= len(dictOfField[keys])


    dictA ={}
    for s in dictOfEdges:
        dictA[s]=len(dictOfEdges[s])


    newDict =dictOfField
    finalD={}
    for s in newDict:
        og = 0
       
        for i in dictA:
            
            for m in range(len(newDict[s])):
                
                if str(i) == str(newDict[s][m]):
                    finalD[s] = og + dictA[i]
                    og = og + dictA[i]
    data = finalD
    dfTotal = pd.DataFrame.from_dict(data,orient='index',columns=['Total New Members'])
    
    return dfTotal                


def show_new_members(G,dictOfField,faculty_path):
    LL = name_with_pid(faculty_path)
    df_edge = nx.to_pandas_edgelist(G)
    for index, row in df_edge.iterrows():
        for i in LL:
            if str(row[1]) == str(i[0]):
                row[1] = i[1]
    for index, row in df_edge.iterrows():
        for i in dictOfField:
            for g in dictOfField[i]:
          
                if str(row[0]) == str(g):
                
                    row[0] = str(i)
            
    df_edges = df_edge.rename(columns={'source': 'Department','target':'Name'})
    
    return df_edges
