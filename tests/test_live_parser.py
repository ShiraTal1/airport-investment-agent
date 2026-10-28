import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agent.data import parse_faa_status  # noqa: E402

SAMPLE = b"""<AIRPORT_STATUS_INFORMATION><Update_Time>Wed Oct 28 09:20:40 2026 GMT</Update_Time>
<Delay_type><Name>Ground Delay Programs</Name><Ground_Delay_List><Ground_Delay><ARPT>SFO</ARPT>
<Reason>low ceilings</Reason><Avg>1 hour and 5 minutes</Avg><Max>2 hours</Max></Ground_Delay></Ground_Delay_List></Delay_type>
<Delay_type><Name>General Arrival/Departure Delay Info</Name><Arrival_Departure_Delay_List><Delay><ARPT>BOS</ARPT>
<Reason>WX:Wind</Reason><Arrival_Departure Type="Arrival"><Min>16 minutes</Min><Max>30 minutes</Max><Trend>Increasing</Trend>
</Arrival_Departure></Delay></Arrival_Departure_Delay_List></Delay_type>
<Delay_type><Name>Airport Closures</Name><Airport_Closure_List><Airport><ARPT>SNA</ARPT><Reason>!SNA AD AP CLSD</Reason>
<Start>Oct 27 at 06:30 UTC.</Start><Reopen>Oct 31 at 13:15 UTC.</Reopen></Airport></Airport_Closure_List></Delay_type>
</AIRPORT_STATUS_INFORMATION>"""


def test_parse():
    out = parse_faa_status(SAMPLE)
    by = {e["airport"]: e for e in out["events"]}
    assert out["ok"] and set(by) == {"SFO", "BOS", "SNA"}
    assert by["SFO"]["detail"]["Reason"] == "low ceilings"
    assert by["BOS"]["detail"]["Arrival_Departure.Max"] == "30 minutes"
    assert by["SNA"]["type"] == "Airport Closures"


if __name__ == "__main__":
    test_parse()
    print("PASS test_parse")
