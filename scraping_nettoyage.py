# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 03:27:06 2024

@author: Baptiste
"""

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.common.action_chains import ActionChains
from bs4 import BeautifulSoup
import requests
import time
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords_fr = set(stopwords.words('french'))
from nltk.tokenize import word_tokenize
from collections import Counter
from spellchecker import SpellChecker



driver = webdriver.Chrome()
driver.get("https://www.opinion-assurances.fr/tous-les-assureurs.html")
code_page = driver.page_source
soup = BeautifulSoup(code_page, 'html.parser')
div = soup.find('div', class_='col-lg-8 oa_allCarriers')
liens = [a.get('href') for a in div.find_all('a', href=True)]
liens= [item for item in liens if item.count("/") != 2]

data = []
    
for link in liens:
    try:
        link=link[:-5]
        i=1
        driver.get("https://www.opinion-assurances.fr"+link+'-page'+str(i)+'.html')
        code_page = driver.page_source
        soup = BeautifulSoup(code_page, 'html.parser')
        elements = soup.find_all(class_='row oa_reactionBlock')
        for bloc in elements:
            nom_profil = bloc.find(itemprop="author").text if bloc.find(itemprop="author") else "Inconnu"
            date_publication = bloc.find(class_="oa_date").text.strip().replace('Avis publié le ', '') if bloc.find(class_="oa_date") else "Date Inconnue"
            contenu_texte = bloc.find(class_="oa_reactionText").text.strip() if bloc.find(class_="oa_reactionText") else "Contenu Inconnu"
            nombre_etoiles_actives = len(bloc.find_all("i", class_="fas fa-star active")) /3
    
            data.append({
                'Nom du Profil': nom_profil,
                'Date de Publication': date_publication,
                'Contenu du Texte': contenu_texte,
                'Nombre d\'Étoiles Actives': nombre_etoiles_actives,
                'Assurance': link
            })
    
        i+=1
        time.sleep(4)
        while str(i-1) in driver.current_url:
            try:
                driver.get("https://www.opinion-assurances.fr"+link+'-page'+str(i)+'.html')
                code_page = driver.page_source
                soup = BeautifulSoup(code_page, 'html.parser')
                elements = soup.find_all(class_='row oa_reactionBlock')
                for bloc in elements:
                    nom_profil = bloc.find(itemprop="author").text if bloc.find(itemprop="author") else "Inconnu"
                    date_publication = bloc.find(class_="oa_date").text.strip().replace('Avis publié le ', '') if bloc.find(class_="oa_date") else "Date Inconnue"
                    contenu_texte = bloc.find(class_="oa_reactionText").text.strip() if bloc.find(class_="oa_reactionText") else "Contenu Inconnu"
                    nombre_etoiles_actives = len(bloc.find_all("i", class_="fas fa-star active")) /3
        
                    data.append({
                        'Nom du Profil': nom_profil,
                        'Date de Publication': date_publication,
                        'Contenu du Texte': contenu_texte,
                        'Nombre d\'Étoiles Actives': nombre_etoiles_actives,
                        'Assurance': link
                    })
        
                i+=1
                time.sleep(4)
    
            except:
                break
    except:
        pass
    

"""---------------------------------------------"""


df = pd.DataFrame(data)
df=df.drop_duplicates()
df['date de l\'expérience'] = df['Date de Publication'].str.extract(r'suite à une expérience en (.*)')
df['Date de Publication'] = df['Date de Publication'].str[:10]
df['Assurance'] = df['Assurance'].str.replace("/", "")
df['Assurance'] = df['Assurance'].str.replace("-", " ")
df['Assurance'] = df['Assurance'].str.replace("assureur ", "")


def creatoken(text):
    if isinstance(text, str):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalpha()]
        tokens = [word.lower() for word in tokens]
        tokens = [word for word in tokens if word not in stopwords_fr ]
        tokens = [word for word in tokens if len(word) > 2]
        return tokens
    else:
        return []

df['token'] = df['Contenu du Texte'].apply(lambda x : creatoken(x))

# frequense mot
all_words = [word for tokens in df['token'] for word in tokens]
word_freq = Counter(all_words)


# correction
spell = SpellChecker(language='fr')
def correct_spelling(text):
    if text is None or not isinstance(text, str):
        return ""
    corrected_words = []
    for word in text.split():
        corrected_word = spell.correction(word)
        if corrected_word is None:
            corrected_word = word
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)
df['texte_corrige'] = df['Contenu du Texte'].apply(correct_spelling)


