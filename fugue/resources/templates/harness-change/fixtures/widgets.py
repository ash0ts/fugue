def create_widget(request):
    widget = {"name": request["name"]}
    return widget, 202
