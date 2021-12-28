from bs4 import BeautifulSoup, SoupStrainer
import requests

urls = ["https://www.giantbomb.com/ratchet-clank/3025-411/characters/",
        "https://www.giantbomb.com/ratchet-clank/3025-411/characters/?page=2"]

def print_characters(url):
    soup = get_soup(url)
    chars = soup.find_all('h3', {"class":"title"})
    for ch in chars:
        print("\"" + ch.getText() + "\",")

def get_soup(url):
    page = requests.get(url)
    if not page:
        raise RuntimeError("Couldn't get the page " + url)
    return BeautifulSoup(page.text, "lxml")

#for url in urls:
#    print_characters(url)

url = "https://ratchetandclank.fandom.com/wiki/Solana_Galaxy"
def print_planets(url):
    soup = get_soup(url)
    tables = soup.findAll('table', {"class": "article-table"})
    table = tables[1]
    #print("table", table)
    for i, row in enumerate(table.tbody.findAll('tr')):
        # Skip the header of the table
        if i == 0:
            continue
        #print("row", row)
        first_column = row.findAll('td')[0].contents[0]
        #print("first", first_column)
        if not isinstance(first_column, str):
            planet = first_column.getText()
            print("\"" + planet + "\",")

#print_planets(url)

url = "https://ratchetandclank.fandom.com/wiki/Category:Weapons_in_A_Crack_in_Time"
def print_weapons(url):
    soup = get_soup(url)
    weapons = soup.find_all("a", {"class": "category-page__member-link"})
    for weapon in weapons:
        print("\"" + weapon.getText() + "\",")

print_weapons(url)

