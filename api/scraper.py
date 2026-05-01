"""
Web Scraper for nbadraft.net

Scrapes player scouting reports from nbadraft.net
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional
import logging
import re
import time

logger = logging.getLogger(__name__)


def scrape_player_scouting_report(player_name: str) -> Optional[Dict[str, Any]]:
    """
    Scrape scouting report for a player from nbadraft.net

    Args:
        player_name: Name of the player to search for

    Returns:
        Dictionary with scouting report data or None if not found
    """
    try:
        # Search for player
        player_url = search_player(player_name)

        if not player_url:
            logger.warning(f"Could not find player: {player_name}")
            return None

        # Scrape player page
        scouting_data = scrape_player_page(player_url)

        return scouting_data

    except Exception as e:
        logger.error(f"Error scraping player {player_name}: {e}")
        return None


def search_player(player_name: str) -> Optional[str]:
    """
    Search for player on nbadraft.net and return their profile URL

    Args:
        player_name: Name of player to search

    Returns:
        URL of player's profile page or None if not found
    """
    try:
        # Normalize player name for URL
        search_name = player_name.lower().replace(" ", "-")

        # Try direct URL first (most common pattern)
        base_url = "https://www.nbadraft.net/players"
        player_url = f"{base_url}/{search_name}"

        response = requests.get(player_url, timeout=10)

        if response.status_code == 200:
            logger.info(f"Found player at: {player_url}")
            return player_url

        # If direct URL doesn't work, try search
        logger.info(f"Direct URL failed, trying search for: {player_name}")
        search_url = "https://www.nbadraft.net/search"
        search_response = requests.get(
            search_url, params={"q": player_name}, timeout=10
        )

        if search_response.status_code == 200:
            soup = BeautifulSoup(search_response.content, "html.parser")

            # Look for player link in search results
            player_links = soup.find_all("a", href=re.compile(r"/players/"))

            if player_links:
                # Return first match
                first_link = player_links[0]["href"]
                full_url = f"https://www.nbadraft.net{first_link}"
                logger.info(f"Found player via search: {full_url}")
                return full_url

        logger.warning(f"Could not find player: {player_name}")
        return None

    except Exception as e:
        logger.error(f"Error searching for player {player_name}: {e}")
        return None


def scrape_player_page(url: str) -> Dict[str, Any]:
    """
    Scrape scouting report from player's nbadraft.net page

    Args:
        url: URL of player's profile page

    Returns:
        Dictionary with scouting data
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, "html.parser")

        # Initialize data dictionary
        data = {
            "Strengths": "",
            "Weaknesses": "",
            "overall": None,
            "Athleticism": None,
            "Size": None,
            "Defense": None,
            "Strength": None,
            "Quickness": None,
            "Leadership": None,
            "JumpShot": None,
            "NBAReady": None,
        }

        # Extract strengths and weaknesses
        # Try multiple methods to find the scouting report text

        # Method 1: Look for text containing "Strengths:" and "Weaknesses:"
        page_text = soup.get_text()

        # Find strengths
        strengths_match = re.search(
            r"Strengths?:\s*(.+?)(?=Weaknesses?:|Outlook:|$)",
            page_text,
            re.DOTALL | re.I,
        )
        if strengths_match:
            data["Strengths"] = strengths_match.group(1).strip()

        # Find weaknesses
        weaknesses_match = re.search(
            r"Weaknesses?:\s*(.+?)(?=Outlook:|Notes:|$)", page_text, re.DOTALL | re.I
        )
        if weaknesses_match:
            data["Weaknesses"] = weaknesses_match.group(1).strip()

        # Extract numerical grades
        # Look for grade elements (this varies by nbadraft.net's current structure)
        grades = extract_grades(soup)
        data.update(grades)

        # Log what was extracted
        logger.info(f"Successfully scraped scouting report from {url}")
        logger.info(f"Extracted strengths length: {len(data['Strengths'])} chars")
        logger.info(f"Extracted weaknesses length: {len(data['Weaknesses'])} chars")
        logger.info(f"Extracted grades: {grades}")

        return data

    except Exception as e:
        logger.error(f"Error scraping player page {url}: {e}")
        raise


def extract_section_text(header_element) -> str:
    """
    Extract text from section following a header element

    Args:
        header_element: BeautifulSoup element for section header

    Returns:
        Extracted text
    """
    text_parts = []

    # Get next siblings until we hit another header
    for sibling in header_element.next_siblings:
        if sibling.name in ["h1", "h2", "h3", "h4", "h5"]:
            break
        if sibling.name in ["p", "div", "ul", "li"]:
            text = sibling.get_text(strip=True)
            if text:
                text_parts.append(text)

    return " ".join(text_parts)


def extract_grades(soup: BeautifulSoup) -> Dict[str, Optional[float]]:
    """
    Extract numerical grades from player page

    Args:
        soup: BeautifulSoup object of player page

    Returns:
        Dictionary of grade values
    """
    grades = {
        "overall": None,
        "Athleticism": None,
        "Size": None,
        "Defense": None,
        "Strength": None,
        "Quickness": None,
        "Leadership": None,
        "JumpShot": None,
        "NBAReady": None,
    }

    # Extract overall rating (same method as training script)
    try:
        overall_div = soup.find("div", class_="overall")
        if overall_div:
            value_span = overall_div.find("span", class_="value")
            if value_span:
                grades["overall"] = float(value_span.text.strip())
                logger.debug(f"Extracted overall: {grades['overall']}")
    except (AttributeError, ValueError) as e:
        logger.debug(f"Could not extract overall grade: {e}")

    # Extract attribute scores (same method as training script)
    try:
        player_attr_obj = soup.find("div", class_="player-attributes")
        if player_attr_obj:
            attr_values = player_attr_obj.find_all(
                "div", class_="div-table-cell attribute-value"
            )
            attr_names = player_attr_obj.find_all(
                "div", class_="div-table-cell attribute-name"
            )

            logger.debug(f"Found {len(attr_names)} attributes")

            for name_elem, value_elem in zip(attr_names, attr_values):
                attr_name = name_elem.text.strip().replace(" ", "")
                attr_value = value_elem.text.strip()

                logger.debug(f"Processing attribute: {attr_name} = {attr_value}")

                # Map HTML attribute names to our keys
                # Remove spaces for comparison
                attr_name_clean = attr_name.replace(" ", "").lower()

                if attr_name_clean == "athleticism":
                    grades["Athleticism"] = float(attr_value)
                elif attr_name_clean == "size":
                    grades["Size"] = float(attr_value)
                elif attr_name_clean == "defense":
                    grades["Defense"] = float(attr_value)
                elif attr_name_clean == "strength":
                    grades["Strength"] = float(attr_value)
                elif attr_name_clean == "quickness":
                    grades["Quickness"] = float(attr_value)
                elif attr_name_clean == "leadership":
                    grades["Leadership"] = float(attr_value)
                elif attr_name_clean in ["jumpshot", "jump shot"]:
                    grades["JumpShot"] = float(attr_value)
                elif attr_name_clean in ["nbaready", "nba ready"]:
                    grades["NBAReady"] = float(attr_value)

    except (AttributeError, ValueError) as e:
        logger.debug(f"Could not extract attribute grades: {e}")

    logger.info(f"Extracted grades: {grades}")
    return grades
