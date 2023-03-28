curl --location --request POST 'http://localhost:8000/dt_translate_nllb' \
--header 'Content-Type: application/json' \
--data-raw '{
    "sourceLangCode_flores": "en",
    "targetLangCode_flores": "es",
    "textArray": [
        "This is a test sentence. What time is it?\n\nTime for hotdogs.",
        "Here is another test sentence. Peter likes hotdogs.\n\nBut he cannot eat them, because he is on a diet.",
        "This sentence contains two parts. The first part ends here.",
        "The second part of the sentence starts here,\nand ends here.",
        "This sentence has a question mark at the end?",
        "This one has an exclamation point at the end!",
        "This is a really long sentence that goes on and on, and is difficult to read.",
        "This is a short sentence.",
        "This one has a comma, but is otherwise simple.",
        "This sentence contains some numbers: 12345 and 67890."
    ]
}'
