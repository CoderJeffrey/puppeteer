from urlextract import URLExtract

extractor = URLExtract()

print('Type \'exit\' to terminate')
while True:
    txt = input('text: ')
    if txt == 'exit':
        break
    urls = extractor.find_urls(txt)
    print('found urls: {}'.format(str(urls)))
