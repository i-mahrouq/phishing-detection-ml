"""Local Streamlit interface for the frozen V1 predictors; no dataset access."""
from pathlib import Path

import streamlit as st

from src.features.url_features import URLFeatures
from src.inference.predict import Predictor

ROOT = Path(__file__).resolve().parent


@st.cache_resource
def predictor(root: str) -> Predictor:
    """Cache only trusted models, never user messages or predictions."""
    return Predictor(Path(root))


def show_result(result: dict) -> None:
    """Render a model label without implying calibrated confidence or URL safety."""
    phishing = result['prediction'] == 'PHISHING'
    if phishing:
        st.error('Prediction: Phishing')
    else:
        st.success('Prediction: Legitimate')
    st.caption('A model prediction is not a safety verdict. False positives and missed attacks are possible.')
    with st.expander('Model details'):
        st.write('Model:', result['model'])
        st.write('Raw decision score:', result['score'])
        st.write('Saved threshold:', result['threshold'])
        st.code('phishing if score >= threshold', language='text')
        if result['score_type'] == 'probability':
            st.write('This is an uncalibrated Random Forest class estimate, not a confidence percentage.')
        else:
            st.write('This is an uncalibrated SVM margin, not a probability.')


def main() -> None:
    st.set_page_config(page_title='Phishing Detection — Machine Learning', page_icon='🛡️', layout='centered')
    st.title('Phishing Detection — Machine Learning')
    st.write('Explore two pretrained V1 classifiers: email content and lexical URL structure. '
             'Inputs stay in this local application; URLs are never visited.')
    st.warning('Research demonstration. Predictions may be incorrect. URL source-format bias and historical '
               'email label/source limitations prevent production safety claims.')
    email_tab, url_tab = st.tabs(['Email detection', 'URL detection'])
    with email_tab:
        st.subheader('Email Phishing Detection')
        with st.form('email_form'):
            subject = st.text_input('Subject (optional)', max_chars=5000)
            body = st.text_area('Email body', height=200, max_chars=100000,
                                placeholder='Paste the email text here. HTML is treated as input text, never rendered.')
            submitted = st.form_submit_button('Analyze Email', type='primary')
        if submitted:
            try:
                show_result(predictor(str(ROOT)).predict_email(subject, body))
            except (ValueError, OSError, KeyError, AssertionError, ImportError) as exc:
                st.error(f'Cannot analyze email: {exc}')
    with url_tab:
        st.subheader('URL Phishing Detection')
        with st.form('url_form'):
            url = st.text_input('URL', max_chars=65536, placeholder='https://example.com/')
            submitted = st.form_submit_button('Analyze URL', type='primary')
        if submitted:
            try:
                result = predictor(str(ROOT)).predict_url(url)
                show_result(result)
                with st.expander('13 lexical URL features'):
                    st.dataframe(URLFeatures().transform([url]), hide_index=True)
            except (ValueError, OSError, KeyError, AssertionError, ImportError) as exc:
                st.error(f'Cannot analyze URL: {exc}')
    st.caption('No datasets or training required. No external API calls or persistent input logging.')


if __name__ == '__main__':
    main()
