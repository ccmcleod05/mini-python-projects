import openai
import requests
import os
from dotenv import load_dotenv
load_dotenv()

def generate_text(key):
    openai.api_key = key

    url_transactions = f"http://{os.getenv('HOST_GET')}:{os.getenv('PORT')}/api/transactions" # good
    transactions_json = requests.get(url_transactions)

    url_identity = f"http://{os.getenv('HOST_GET')}:{os.getenv('PORT')}/api/identity" # good
    identity_json = requests.get(url_identity)

    url_balance = f"http://{os.getenv('HOST_GET')}:{os.getenv('PORT')}/api/balance" #good
    balance_json = requests.get(url_balance)

    url_investment_holdings = f"http://{os.getenv('HOST_GET')}:{os.getenv('PORT')}/api/holdings" #good
    investment_holdings_json = requests.get(url_investment_holdings)

    url_liabilities = f"http://{os.getenv('HOST_GET')}:{os.getenv('PORT')}/api/liabilities" #good
    liabilities_json = requests.get(url_liabilities)

    city = identity_json.json()['identity'][0]['owners'][0]['addresses'][0]['data']['city']
    country = identity_json.json()['identity'][0]['owners'][0]['addresses'][0]['data']['country']
    income = None

    i = 0
    transactions_string = ''
    for transaction in transactions_json.json()['latest_transactions']:
        amount = transaction['amount']
        transactions_string += transaction['name'] + ' - '
        transactions_string += f'${amount}'
        transactions_string += '\n'

    accounts_string = ''
    for account in balance_json.json()['accounts']:
        account_name = account['name']
        balance = account['balances']['current']

        accounts_string += f'My {account_name} account balance is: {balance}'
        accounts_string += '\n'

    investments_string = ''
    for i in range(0, len(investment_holdings_json.json()['holdings']['holdings'])-1):

        investment_quantity = investment_holdings_json.json()['holdings']['holdings'][i]['quantity']
        investment_close_price = investment_holdings_json.json()['holdings']['securities'][i]['close_price']
        investment_name = investment_holdings_json.json()['holdings']['securities'][i]['name']
        if investment_close_price != None and investment_quantity != None:
            investment_close_price = float(investment_close_price)
            investment_quantity = float(investment_quantity)
            investment_total = investment_close_price * investment_quantity
        investments_string += f'{investment_name} - '
        investments_string += f'${investment_total}'
        investments_string += '\n'

    long_term_plan = f"""
        What are some long term plans for what I should do with my money based on the following information:

        I live in {city}, {country}

        My recent transaction history is:
        {transactions_string}

        My current account balances are:
        {accounts_string}
        The type of account is in the name

        My current investment holdings are: 
        {investments_string}

        Format so that there are only new lines between the numbered responses and no where else
    """

    short_term_plan = f"""
        What are some short term plans for what I should do with my money based on the following information:

        I live in {city}, {country}

        My recent transaction history is:
        {transactions_string}

        My current account balances are:
        {accounts_string}
        The type of account is in the name

        My current investment holdings are: 
        {investments_string}

        Format so that there are only new lines between the numbered responses and no where else
    """

    daily_tips = f"""
        What are some specific tips for what I should do with my money based on the following information:

        I live in {city}, {country}

        My recent transaction history is:
        {transactions_string}

        My current account balances are:
        {accounts_string}
        The type of account is in the name

        My current investment holdings are: 
        {investments_string}

        Format so that there are only new lines between the numbered responses and no where else
    """

    response_long = openai.Completion.create(
        model="text-davinci-003",
        prompt = "<|endoftext|>"+long_term_plan+"\n--\nLabel:",
        temperature=.7,
        max_tokens=188,
        top_p=0,
        logprobs=10
    )
    response_short = openai.Completion.create(
        model="text-davinci-003",
        prompt = "<|endoftext|>"+short_term_plan+"\n--\nLabel:",
        temperature=.7,
        max_tokens=188,
        top_p=0,
        logprobs=10
    )
    response_tips = openai.Completion.create(
        model="text-davinci-003",
        prompt = "<|endoftext|>"+daily_tips+"\n--\nLabel:",
        temperature=.7,
        max_tokens=188,
        top_p=0,
        logprobs=10
    )

    long = response_long.to_dict()["choices"][0]["text"]
    short = response_short.to_dict()["choices"][0]["text"]
    tips = response_tips.to_dict()["choices"][0]["text"]

    short_term_plan_1 = short[short.find('1.'):short.find('\n', short.find('1.'))]
    short_term_plan_2 = short[short.find('2.'):short.find('\n', short.find('2.'))]
    long_term_plan_1 = long[long.find('1.'):long.find('\n', long.find('1.'))]
    long_term_plan_2 = long[long.find('2.'):long.find('\n', long.find('2.'))]
    daily_tip_1 = tips[tips.find('1.'):tips.find('\n', tips.find('1.'))]
    daily_tip_2 = tips[tips.find('2.'):tips.find('\n', tips.find('2.'))]

    goal_dict = {'st1': short_term_plan_1, 'st2': short_term_plan_2, 'lt1': long_term_plan_1, 'lt2': long_term_plan_2, 't1': daily_tip_1, 't2': daily_tip_2}
    print(goal_dict)
    return goal_dict

generate_text(os.getenv('OPENAI_API_KEY'))