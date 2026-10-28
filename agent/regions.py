"""Region and metro-area definitions used to scope rankings.

Regions follow U.S. Census divisions (a neutral, well-known standard), plus a few
investor-friendly groupings. Metro groups map colloquial names ("LA", "Bay Area")
to the airports an analyst would usually mean.
"""

REGIONS = {
    "New England": ["CT", "ME", "MA", "NH", "RI", "VT"],
    "Mid-Atlantic": ["NJ", "NY", "PA"],
    "East North Central": ["IL", "IN", "MI", "OH", "WI"],
    "West North Central": ["IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South Atlantic": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV"],
    "East South Central": ["AL", "KY", "MS", "TN"],
    "West South Central": ["AR", "LA", "OK", "TX"],
    "Mountain": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY"],
    "Pacific": ["AK", "CA", "HI", "OR", "WA"],
    # Convenience groupings
    "Northeast": ["CT", "ME", "MA", "NH", "RI", "VT", "NJ", "NY", "PA"],
    "Midwest": ["IL", "IN", "MI", "OH", "WI", "IA", "KS", "MN", "MO", "NE", "ND", "SD"],
    "South": ["DE", "DC", "FL", "GA", "MD", "NC", "SC", "VA", "WV", "AL", "KY", "MS",
              "TN", "AR", "LA", "OK", "TX"],
    "West": ["AZ", "CO", "ID", "MT", "NV", "NM", "UT", "WY", "AK", "CA", "HI", "OR", "WA"],
    "West Coast": ["CA", "OR", "WA"],
    "Contiguous US": None,  # handled in code: all states except AK, HI
    "US": None,             # handled in code: everything
}

METRO_GROUPS = {
    "los angeles area": ["LAX", "BUR", "LGB", "ONT", "SNA"],
    "san francisco bay area": ["SFO", "OAK", "SJC"],
    "new york area": ["JFK", "LGA", "EWR", "HPN", "ISP"],
    "washington dc area": ["DCA", "IAD", "BWI"],
    "chicago area": ["ORD", "MDW"],
    "boston area": ["BOS", "PVD", "MHT"],
    "south florida": ["MIA", "FLL", "PBI"],
    "dallas area": ["DFW", "DAL"],
    "houston area": ["IAH", "HOU"],
}

# Colloquial names -> the airport an analyst most likely means.
# The agent states this assumption explicitly when it uses one.
ALIASES = {
    "la": "LAX", "los angeles": "LAX", "lax": "LAX",
    "santa ana": "SNA", "orange county": "SNA", "john wayne": "SNA",
    "sf": "SFO", "san francisco": "SFO",
    "anchorage": "ANC",
    "boston": "BOS", "logan": "BOS",
    "nyc": "JFK", "new york": "JFK",
    "chicago": "ORD", "dc": "DCA", "washington": "DCA",
    "vegas": "LAS", "las vegas": "LAS",
    "providence": "PVD", "hartford": "BDL", "manchester": "MHT",
    "portland maine": "PWM", "burlington": "BTV",
}


def states_for_region(name: str):
    """Return (canonical_region_name, list_of_states or None) for a region name."""
    if not name:
        return "US", None
    key = name.strip().lower()
    for canon, states in REGIONS.items():
        if canon.lower() == key:
            return canon, states
    # Loose matches
    loose = {"new england": "New England", "northeast": "Northeast", "midwest": "Midwest",
             "south": "South", "west": "West", "west coast": "West Coast",
             "mountain west": "Mountain", "pacific northwest": None}
    if key in loose and loose[key]:
        return loose[key], REGIONS[loose[key]]
    if key == "pacific northwest":
        return "Pacific Northwest", ["OR", "WA"]
    return None, None
