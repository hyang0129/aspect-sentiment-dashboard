import requests
import bs4

def get_reviews(URL):
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}
    web_page = bs4.BeautifulSoup(requests.get(URL, headers=headers).text, "lxml")


    text =  web_page.find_all('script', {'type' : 'application/ld+json'})[0].text

    text = text.replace('&amp' , '&')
    text = text.replace('&apos;', "'")

    d = eval(text)
    reviews = d['review']

    for review in reviews:
        review['text'] = review['description']


    return reviews
