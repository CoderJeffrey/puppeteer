import requests
from urlextract import URLExtract

extractor = URLExtract()

print('Type \'exit\' to terminate')
while True:
    txt = input('text: ')
    if txt == 'exit':
        break
    extract_urls = extractor.find_urls(txt)
    print('extracted urls: {}'.format(str(extract_urls)))
   
    for i, url in enumerate(extract_urls):
        if "http" not in url:
            url = "http://" + url
        try:
            request_response = requests.head(url)
            if request_response.status_code == 404:
                print("{}) {} is valid but not reachable.".format(i, url))
            else:
                print("{}) {} is valid and reachable.".format(i, url))
        except:
            print("{}) is invalid.".format(i, url))
