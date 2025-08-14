import requests
from os import getenv
from dotenv import load_dotenv
# These variables should be know before the call
load_dotenv()
CAL_API_KEY = getenv("CAL_API_KEY")
USERNAME = getenv("USERNAME")
SLUG = '30min' 
PROSPECT_TIMEZONE = 'America/New_York'

'''
The start time of the booking in ISO 8601 format in UTC timezone.
IANA time zone identifier
'''

## get times ##
async def available_times(start_date: str, end_date: str, duration: int = 30) -> dict:
    url = "https://api.cal.com/v2/slots"
    querystring = {
        "start": start_date,
        "end": end_date,
        "username": USERNAME,          
        "timeZone": PROSPECT_TIMEZONE,          
        "eventTypeSlug": SLUG,
        "duration": duration
    }
    headers = {
        "cal-api-version": "2024-09-04",
        "Authorization": f"Bearer {CAL_API_KEY}"  # fixed f-string
    }
    response = requests.get(url, headers=headers, params=querystring)
    response.raise_for_status()  # raises if status != 200
    return response.json()
## book meeting ##
async def book_meeting(name: str, email: str, phone: str, start_time: str):
    url = "https://api.cal.com/v2/bookings"
    payload = {
        "start": start_time,
        "attendee": {
            "language": "en",
            "name": name,
            "timeZone": PROSPECT_TIMEZONE,  # use your global PROSPECT_TIMEZONE constant
            "email": email,
            "phoneNumber": phone
        },
        "username": USERNAME,      # required for booking
        "eventTypeSlug": SLUG     # which event type
    }

    headers = {
        "cal-api-version": "2024-08-13",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {CAL_API_KEY}"  }

    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()

    return response.json()
