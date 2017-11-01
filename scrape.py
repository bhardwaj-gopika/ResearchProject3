
# coding: utf-8

# ## Extracting just the links from the Security home page

# In[126]:

base_url = "https://www.cnet.com"
additional_url = "/topics/security/1169/"

import re
import numpy as np
import requests
from bs4 import BeautifulSoup
from time import sleep

# To keep a count of the number of articles to be scrapped
limit = 0;

next_page = base_url + additional_url

# List to store the links
list_of_links = []

# Change the limit as per requirements
while next_page and limit <= 500:
	
    print(next_page)
    print(limit)
    temp_list_of_links = []
    # Load and extract the content of the page
    page = requests.get(next_page)
    #sleep(120)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find the 'news' links of the page
    for link in soup.find_all('a', href=True):
        if link['href'].startswith('/news/'):
            temp_list_of_links.append(link['href'])
            
    # Save the unique links
    link_list = set(temp_list_of_links)
    
    # Find the length of the list of unique links
    length = len(link_list)
    #print(length)
    
    # Add the links to the final list
    list_of_links.extend(link_list)

    #sleep(120)
    
    # Increment the limit
    limit = limit + length
    
    # Find the links of the Show More page
    next_page = soup.find('a', class_='load-more')
    
    # Change the href to the Show More page link
    if next_page : 
        next_page = base_url + next_page['href']

    


# In[127]:

# Final list with unique links
link_list = set(list_of_links)

# Remove the lone '/news'/ link
link_list.remove('/news/')

# Converting the set into a list
link_list = list(link_list)


# ## Extracting the data from each link

# In[128]:
i = 0
all_articles = []
for item in link_list:
    
    new_page = base_url + item
    print(new_page)
    page = requests.get(new_page)
    i = i + 1
    soup = BeautifulSoup(page.content, 'html.parser')

    #sleep(120)
    
    article = []
    article_title = soup.title.text
    article.append(article_title)

    #print(soup.prettify())

    article_content = []
    content = soup.find("div", {"class":"col-7 article-main-body row"}).findAll('p')

    # Writing the content found in the list in its text form
    for item in content:
        article_content.append(item.text)

    # Joining the list elements to form a proper paragraph
    article_content = " ".join(article_content)

    article.append(article_content)
    all_articles.append(article)

# In[129]:

import pandas as pd
df = pd.DataFrame()
df = df.append(all_articles)
df.to_csv('data.csv',mode = 'a', encoding='utf-8')


# In[1181]:



