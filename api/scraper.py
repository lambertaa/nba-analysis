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
        # Look for sections with these headings
        strengths_section = soup.find(
            ["h3", "h4", "strong"], string=re.compile(r"Strengths?", re.I)
        )
        if strengths_section:
            # Get text from following paragraph or div
            strengths_text = extract_section_text(strengths_section)
            data["Strengths"] = strengths_text

        weaknesses_section = soup.find(
            ["h3", "h4", "strong"], string=re.compile(r"Weaknesses?", re.I)
        )
        if weaknesses_section:
            weaknesses_text = extract_section_text(weaknesses_section)
            data["Weaknesses"] = weaknesses_text

        # Extract numerical grades
        # Look for grade elements (this varies by nbadraft.net's current structure)
        grades = extract_grades(soup)
        data.update(grades)

        logger.info(f"Successfully scraped scouting report from {url}")
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

    try:
        # nbadraft.net uses various structures for grades
        # Try to find grade elements by common patterns

        # Pattern 1: Look for grade bars or progress bars
        grade_elements = soup.find_all(
            ["div", "span"], class_=re.compile(r"grade|rating|score", re.I)
        )

        for element in grade_elements:
            # Try to extract grade name and value
            grade_text = element.get_text(strip=True)

            # Look for patterns like "Athleticism: 85" or "Defense 7.5"
            match = re.search(
                r"(Athleticism|Size|Defense|Rebounding|Jump\s*Shot|NBA\s*Ready|Overall)[:\s]+(\d+\.?\d*)",
                grade_text,
                re.I,
            )

            if match:
                grade_name = match.group(1).replace(" ", "")
                grade_value = float(match.group(2))

                # Normalize grade name
                if "athleticism" in grade_name.lower():
                    grades["Athleticism"] = grade_value
                elif "size" in grade_name.lower():
                    grades["Size"] = grade_value
                elif "defense" in grade_name.lower():
                    grades["Defense"] = grade_value
                elif "rebound" in grade_name.lower():
                    grades["Rebounding"] = grade_value
                elif "jump" in grade_name.lower() or "shot" in grade_name.lower():
                    grades["JumpShot"] = grade_value
                elif "nba" in grade_name.lower() and "ready" in grade_name.lower():
                    grades["NBAReady"] = grade_value
                elif "overall" in grade_name.lower():
                    grades["overall"] = grade_value

        # Pattern 2: Look for data attributes
        for grade_name in grades.keys():
            attr_name = f"data-{grade_name.lower()}"
            element = soup.find(attrs={attr_name: True})
            if element:
                try:
                    grades[grade_name] = float(element[attr_name])
                except (ValueError, KeyError):
                    pass

        logger.info(f"Extracted grades: {grades}")

    except Exception as e:
        logger.warning(f"Error extracting grades: {e}")

    return grades
