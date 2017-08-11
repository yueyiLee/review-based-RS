# -*- coding: UTF-8 -*-
import url_manager,html_downloader,html_outputer,html_parser
class SpiderMain(object):
    def __init__(self):
        self.urls = url_manager.UrlManager()
        self.downloader = html_downloader.HtmlDownloader()
        self.parser = html_parser.HtmlParser()
        self.outputer = html_outputer.HtmlOutputer()
    def craw(self,root_url):
        self.urls.add_new_url(root_url)
        count = 1
        while self.urls.has_new_url():
            new_url = self.urls.get_new_url()
            print("craw %d:%s"%(count, new_url))
            html_cont = self.downloader.downloder(new_url)
            new_urls, new_data = self.parser.parser(new_url, html_cont)
            self.urls.add_new_urls(new_urls)
            self.outputer.collect_data(new_data)
            try:
                if count == 1000:
                    break
                count += 1
            except:
                print("failed")



        self.outputer.output_html()



if __name__ == "__main__":
    root_url = "http://baike.baidu.com/item/Python"
    obj_spider = SpiderMain()
    obj_spider.craw(root_url)