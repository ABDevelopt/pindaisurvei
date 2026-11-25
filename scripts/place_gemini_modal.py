with open('static/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

marker = 'id="gemini-key-modal"'
first_pos = text.find(marker)
main_pos = text.find('<main>')

if first_pos != -1 and first_pos < main_pos:
    print('Modal is already placed globally')
else:
    m_start = text.find('<!-- Gemini API Key Configuration Modal -->')
    m_end = text.find('loadGeminiStatus();', m_start)
    if m_start != -1 and m_end != -1:
        modal_html = text[m_start:m_end].rstrip()
        toast_idx = text.find('<div id="toast-container"></div>')
        if toast_idx != -1:
            text = text[:toast_idx] + modal_html + '\n\n  ' + text[toast_idx:]
            with open('static/index.html', 'w', encoding='utf-8') as f:
                f.write(text)
            print('gemini-key-modal successfully placed globally!')
