lowercase_list = [chr(i) for i in range(97, 123)]
Uppercase_list = [chr(i) for i in range(65, 90)]
Whitespace = [" "]
allowedchars = lowercase_list + Uppercase_list + Whitespace

def removepunctuation(s):
    output = ""
    for chars in s:
        for char in allowedchars:
            if chars == char:
                output += char
    print(output)

if __name__ == "__main__":
    removepunctuation("Let's try, Mike!🛸")