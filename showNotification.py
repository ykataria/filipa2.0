from win10toast import ToastNotifier

def showNotif(title, msg, duration):
    toaster = ToastNotifier()
    toaster.show_toast(
        title=title,
        msg=msg,
        icon_path=None,
        duration=duration,
        threaded=True)