import os
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import concurrent.futures
import bs4

folder_name = 'images'
if not os.path.isdir(folder_name):
    os.makedirs(folder_name)

def download_image(url, folder_name, num):
    response = requests.get(url)
    if response.status_code == 200:
        with open(os.path.join(folder_name, f"Real-Life_{num}.jpg"), 'wb') as file:
            file.write(response.content)

def scrape_images(search_url, folder_name, start_index):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1920x1080")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--no-sandbox")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    driver.get(search_url)

    time.sleep(20)
    driver.execute_script("window.scrollTo(0, 0)")
    time.sleep(1)

    page_html = driver.page_source
    pageSoup = bs4.BeautifulSoup(page_html, 'html.parser')
    containers = pageSoup.findAll('div', {'class': 'eA0Zlc WghbWd FnEtTd mkpRId m3LIae RLdvSe qyKxnc ivg-i PZPZlf GMCzAd'})
    len_containers = len(containers)
    print(f"Found {len_containers} images on {search_url}")

    clicked_images = 0
    start_time = time.time()

    for i in range(1, 901):
        if i % 25 == 0:
            continue
        else:
            try:
                first_image = driver.find_element(By.XPATH, f'//*[@id="rso"]/div/div/div[1]/div/div/div[{i}]')
                first_image.click()
                clicked_images += 1
                print(f"Clicked image result {i} on {search_url}")
                time.sleep(2)
                image_element = driver.find_element(By.XPATH, f'//*[@id="Sva75c"]/div[2]/div[2]/div[2]/div[2]/c-wiz/div/div/div/div/div[3]/div[1]/a/img[1]')
                image_url = image_element.get_attribute('src')
                print(image_url)
            except Exception as e:
                print(f"Error clicking image result {i} on {search_url}: {e}")
            try:
                download_image(image_url, folder_name, start_index + i)
                print(f"Downloaded element {i} out of {len_containers + 1} total on {search_url}. URL: {image_url}")
            except:
                print(f"Couldn't download an image {i} on {search_url}, continuing to the next one")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"Clicked {clicked_images} images on {search_url}")
    print(f"Total time taken on {search_url}: {total_time} seconds")
    driver.quit()

# URLs to scrape
urls = [
    "https://www.google.com/search?q=Sunset+photography+real+life&sca_esv=5efdf66075fec85d&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIJkyKVvWKpp00_xJ0kyZMxjuy4coQ%3A1716547083928&ei=C25QZrSgOLqfseMP6Pas0AQ&ved=0ahUKEwj00bWZjKaGAxW6T2wGHWg7C0oQ4dUDCBA&uact=5&oq=Sunset+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiHFN1bnNldCBwaG90b2dyYXBoeSByZWFsIGxpZmVI7wRQpAFYpAFwAXgAkAEAmAGhAaABoQGqAQMwLjG4AQPIAQD4AQL4AQGYAgCgAgCYAwCIBgGSBwCgBy0&sclient=gws-wiz-serp",
    "https://www.google.com/search?q=Beach+photography+real+life&sca_esv=00e485a4403845c8&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIJvBSVs64rliowUdTorckxqIgP75g%3A1716547153957&ei=UW5QZtCBOqmhseMPss-lgAs&ved=0ahUKEwiQ7ue6jKaGAxWpUGwGHbJnCbAQ4dUDCBA&uact=5&oq=Beach+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiG0JlYWNoIHBob3RvZ3JhcGh5IHJlYWwgbGlmZUjVAlCKAViKAXABeACQAQCYAZcBoAGXAaoBAzAuMbgBA8gBAPgBAvgBAZgCAKACAJgDAIgGAZIHAKAHLQ&sclient=gws-wiz-serp",
    "https://www.google.com/search?q=Cityscape+photography+real+life&sca_esv=00e485a4403845c8&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIK2_j0Ke9u5QJirVHlaz8fKZnebrw%3A1716547121375&ei=MW5QZoa4Fu3VseMP7fin8Ac&ved=0ahUKEwjGlKOrjKaGAxXtamwGHW38CX4Q4dUDCBA&uact=5&oq=Cityscape+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiH0NpdHlzY2FwZSBwaG90b2dyYXBoeSByZWFsIGxpZmVIkQhQpgJYpgJwAXgAkAEAmAGWAaABlgGqAQMwLjG4AQPIAQD4AQL4AQGYAgGgAgvCAgYQABgIGB6YAwCIBgGSBwExoAct&sclient=gws-wiz-serp",
    "https://www.google.com/search?q=Flowers+photography+real+life&sca_esv=680b12c94771f77f&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIJG1lNWCdkx6cyR63EpkMIIOuHyZw%3A1716536583561&ei=B0VQZrHoIfmZ4-EPrtmSmAw&ved=0ahUKEwjxh7qK5aWGAxX5zDgGHa6sBMMQ4dUDCBA&uact=5&oq=Flowers+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiHUZsb3dlcnMgcGhvdG9ncmFwaHkgcmVhbCBsaWZlSKlIUABYrUdwB3gAkAEAmAHMAqABiiWqAQgwLjI3LjEuMbgBA8gBAPgBAvgBAZgCHKAC8h3CAhAQABiABBixAxhDGIMBGIoFwgIIEAAYgAQYsQPCAgoQABiABBhDGIoFwgIFEAAYgATCAgkQABiABBgYGArCAgYQABgIGB6YAwCSBwg1LjIxLjEuMaAHimU&sclient=gws-wiz-serp",
    "https://www.google.com/search?q=Animals+photography+real+life&sca_esv=680b12c94771f77f&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIJQ4wxfsULAzsG03eNGnPilRBXvUQ%3A1716536594182&ei=EkVQZvnaCqjC4-EPoLyd8Ao&ved=0ahUKEwj5q8KP5aWGAxUo4TgGHSBeB64Q4dUDCBA&uact=5&oq=Animals+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiHUFuaW1hbHMgcGhvdG9ncmFwaHkgcmVhbCBsaWZlSP42UABY_zVwAXgAkAEAmAGYAqABgRyqAQYwLjIyLjG4AQPIAQD4AQL4AQGYAhWgAuwYwgIIEAAYgAQYsQPCAgoQABiABBhDGIoFwgIFEAAYgATCAgYQABgFGB7CAgQQABgewgIGEAAYCBgemAMAkgcGMS4xOS4xoAfnXQ&sclient=gws-wiz-serp",
    "https://www.google.com/search?q=Birds+photography+real+life&sca_esv=680b12c94771f77f&rlz=1C1ONGR_enIN1072IN1072&udm=2&biw=1536&bih=791&sxsrf=ADLYWIKpLkq0LhVjI87grvkXxw3SCNpkZg%3A1716536603306&ei=G0VQZteqEuHz4-EPpP2loAk&ved=0ahUKEwiXpO-T5aWGAxXh-TgGHaR-CZQQ4dUDCBA&uact=5&oq=Birds+photography+real+life&gs_lp=Egxnd3Mtd2l6LXNlcnAiG0JpcmRzIHBob3RvZ3JhcGh5IHJlYWwgbGlmZUiC4I8IUIeTjwhYiN-PCHALeACQAQCYAcgBoAGfJKoBBjAuMjkuMbgBA8gBAPgBAvgBAZgCHaACoBzCAg0QABiABBixAxhDGIoFwgIKEAAYgAQYQxiKBcICCBAAGIAEGLEDwgIFEAAYgATCAgYQABgIGB7CAgQQABgewgIGEAAYBRgemAMAiAYBkgcGNi4yMi4xoAf2aA&sclient=gws-wiz-serp"
     ]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(scrape_images, url, folder_name, (i+1) * 7000) for i, url in enumerate(urls)]
    concurrent.futures.wait(futures)

print("All tasks completed.")
