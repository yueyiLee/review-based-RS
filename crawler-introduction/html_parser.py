# -*- coding: UTF-8 -*-
from bs4 import BeautifulSoup
import re,urlparse
class HtmlParser(object):

    def _get_new_urls(self,page_url, soup):
        new_urls = set()
        links = soup.find_all('a',href=re.compile(r'/item'))
        for link in links:
            new_url = link['href']
            new_full_url = urlparse.urljoin(page_url, new_url)
            new_urls.add(new_full_url)
        return new_urls

    def _get_new_datas(self,page_url, soup):
        new_datas = {}
        #url
        new_datas['url'] = page_url
        #<dd class="lemmaWgt-lemmaTitle-title"><h1>Python</h1>
        title_node = soup.find('dd',class_='lemmaWgt-lemmaTitle-title').find('h1')
        #print(title_node)
        new_datas['title'] = title_node.get_text()
        #<div class="lemma-summary" label-module="lemmaSummary">

        summary_node = soup.find('div',class_="lemma-summary")
        if summary_node is None:
            return
       # print(summary_node)
        new_datas['summary'] = summary_node.get_text()
        return new_datas

    def parser(self,page_url, html_cont):
        if page_url is None or html_cont is None:
            return
        soup = BeautifulSoup(html_cont, 'html.parser', from_encoding='utf-8')
        new_urls = self._get_new_urls(page_url, soup)
        new_data = self._get_new_datas(page_url, soup)
        return new_urls,new_data
