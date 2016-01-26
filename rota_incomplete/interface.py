import requests
import json

baseurl = 'https://mastermind.praetorian.com/'
email = 'chesterholtz@gmail.com'

r = requests.post(baseurl + 'api-auth-token/', data={'email':email})
headers = r.json()
headers['Content-Type'] = 'application/json'

def reset_game():
    reseturl = baseurl + '/reset/'
    r = requests.post(reseturl, headers=headers)
    j = r.json()
    return j

def get_new_level(levelnum):
    levelurl = baseurl + 'level/' + str(levelnum) + '/'
    r = requests.get(levelurl, headers=headers)
    return r.json()

def submit_guess(levelnum, guess):
    levelurl = baseurl + 'level/' + str(levelnum) + '/'
    r = requests.post(levelurl, headers=headers, data=json.dumps({'guess':guess}))
    return r.json()

def get_hash():
    hashurl = baseurl + 'hash/'
    request = requests.get(hashurl, headers=headers)
    json = r.json()
    if 'hash' in json.keys():
        return json['hash']
    else:
        return None

