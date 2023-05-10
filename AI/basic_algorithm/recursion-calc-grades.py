
def calc(grades):
    grade = input()
    if grade == '':
        answer = input("are you sure to exit? -- press enter to exit or any key to not: ")
        if answer == "":
            if len(grades) == 0:
                print("no grades collected")
                calc(grades)
        else:
            calc(grades)
    else:
        try: 
            grades.append(int(grade))
        except ValueError as e:
            print("invalid input, re-enter: ")
        calc(grades)

    return (sum(grades) / len(grades))

print("enter grade (press enter to exit): ")
print(f'grade: {calc([])}')

