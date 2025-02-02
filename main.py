from flask import Flask, Markup, request, render_template
from models import generate_text

#
# def annotate(text):
#     ann_strings = models(text)
#
#     tokens_soup = []
#     comments = {}
#     for i, ann in enumerate(ann_strings):
#         tokens_soup.append([str(ann[0]), int(ann[1])])
#         if ann[2]:
#             comments["comment" + str(i)] = ann[2]
#
#     return tokens_soup, comments


# def annotate_print(text):
#     tokens_soup, comments = annotate(text)
#     result_trial = ''
#     for token in tokens_soup:
#         text = text.replace(token[0], chr(8), 1)
#     flag = False
#     for k, token in enumerate(tokens_soup):
#         entry = ""
#         if token[1] == 1:
#             if "comment" + str(k) in comments:
#                 if flag:
#                     entry += ' '
#                 entry += '<div class="duo"'
#                 entry += ' id="' + "comment" + str(k) + 'link"'
#                 entry += ' onclick="popupbox(event,'
#                 entry += "'" + "comment" + str(k) + "'" + ')">'
#                 flag = True
#             else:
#                 entry += '<div class="duo">'
#         entry += token[0]
#
#         if token[1] == 1:
#             entry += '</div>'
#
#         result_trial += entry
#
#     # text = text.replace("\n", "<br>")
#     result_trial = result_trial.replace("\n", "<br>")
#     # print(result_trial)
#     return result_trial, comments


app = Flask(__name__)


@app.route('/')
def index():
    if request.args:
        text_to_inspect = request.args['text_to_inspect']
        annotation, comments = generate_text(text_to_inspect)
        #print(annotation)
        annotation = Markup(annotation)
        #print(annotation)
        #print(render_template("result.html", annotation=annotation, comments=comments))
        return render_template("result.html", annotation=annotation, comments=comments)
    return render_template("main.html")


if __name__ == '__main__':
    app.run(debug=True)
