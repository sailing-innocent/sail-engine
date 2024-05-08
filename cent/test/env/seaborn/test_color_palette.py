import pytest 
import seaborn as sns

# https://www.practicalpythonfordatascience.com/ap_seaborn_palette
# https://seaborn.pydata.org/tutorial/color_palettes.html
@pytest.mark.current 
def test_color_palette():
    palette = sns.color_palette("RdYlGn")
    print(palette[0])