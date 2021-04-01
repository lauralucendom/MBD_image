#Imports Packages
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import requests
import time 
import urllib.request
import os
import platform

class GOOGLE_IMAGES:
    def __init__(self):
        print("init") # never prints

    def get_new_images(self, search_keyword = [], save_dir = 'nuevas_imagenes', keywords = [' face', ' side face', ' looking up', ' looking down', ' wearning glasses', ' happy face', ' close up'], visible = False):
        if len(search_keyword) == 0:
            print("Se necesitan personas en el parámetro 'search_keyword'.")
        else:
            for s_keyword in search_keyword:
                # Chromedriver
                chromeOptions = webdriver.chrome.options.Options()
                chromeOptions.add_argument("--start-maximized")
                if visible == True : chromeOptions.add_argument("--headless")
                if platform.system() == "Windows":
                    driver = webdriver.Chrome(executable_path="chromedriver.exe", chrome_options=chromeOptions)
                else:
                    driver = webdriver.Chrome(chrome_options=chromeOptions)
                driver.get('https://www.google.es/imghp?hl=en-GB&authuser=0&ogbl')
                time.sleep(1)
                driver.find_element_by_xpath('//*[@id="zV9nZe"]/div').click() # Aceptar las cookies

                # Tratamos de crear las carpetas
                try:
                    os.makedirs(save_dir + '/' + s_keyword)
                except OSError as e:
                    if e.errno != 17:
                        raise   
                    pass
                first = True
                for keyword in keywords:
                    try:
                        pure_keyword = keyword
                        if first == True:
                            input_text = driver.find_element_by_xpath('//*[@id="sbtc"]/div/div[2]/input')
                            first = False
                        else:
                            input_text = driver.find_element_by_xpath('//*[@id="REsRA"]')
                            input_text.clear()
                            time.sleep(2)
                        input_text.send_keys(s_keyword+pure_keyword+' HD')
                        input_text.send_keys(Keys.ENTER)
                        time.sleep(1)

                        for i in range(1,5):
                            try:
                                lnk = driver.find_element_by_xpath('//*[@id="islrg"]/div[1]/div['+str(i)+']/a[1]/div[1]/img').get_attribute('src')
                                urllib.request.urlretrieve(lnk, "nuevas_imagenes/{}/{}_{}.jpg".format(s_keyword, keyword.replace(" ", ""), i))
                            except:
                                pass
                    except:
                        driver.close()
                driver.close()





