"""
A module for Summary Report
"""

from rtichoke.helpers.send_post_request_to_r_rtichoke import send_requests_to_rtichoke_r


def create_summary_report(probs, reals, url_api="http://localhost:4242/"):
    """Create rtichoke Summary Report

    Args:
        probs (_type_): _description_
        reals (_type_): _description_
        url_api (str, optional): _description_. Defaults to "http://localhost:4242/".
    """
    rtichoke_response = send_requests_to_rtichoke_r(
        dictionary_to_send={"probs": probs, "reals": reals},
        url_api=url_api,
        endpoint="create_summary_report",
    )
    print(rtichoke_response.json()[0].keys())
