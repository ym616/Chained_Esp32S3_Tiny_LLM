import os
import json
import random
import string

ERRORS = [
    # General Programming
    "SyntaxError: unexpected EOF while parsing",
    "NullReferenceException: Object reference not set to an instance of an object.",
    "Segmentation fault (core dumped)",
    "TypeError: Cannot read properties of undefined (reading '{var}')",
    "IndentationError: expected an indented block",
    "java.lang.NullPointerException",
    "IndexError: list index out of range",
    "ImportError: No module named '{var}'",
    "AttributeError: '{type}' object has no attribute '{var}'",
    "ReferenceError: {var} is not defined",
    "Error: ENOSPC: no space left on device",
    "MemoryError: Out of memory",
    "StackOverflowError",
    "Compilation failed: 1 error",
    "ModuleNotFoundError: No module named '{var}'",
    "ZeroDivisionError: division by zero",
    "KeyError: '{var}'",
    "ValueError: invalid literal for int() with base 10: '{var}'",
    "FileNotFoundError: [Errno 2] No such file or directory: '{var}'",
    "TimeoutError: Connection timed out",
    "AssertionError: Expected true but got false",
    "RuntimeError: Maximum call stack size exceeded",
    "TypeError: '{type}' object is not callable",
    "UnboundLocalError: local variable '{var}' referenced before assignment",
    "SystemError: unknown opcode",
    "SyntaxError: invalid syntax",
    "NameError: name '{var}' is not defined",
    
    # Arduino Specific
    "avrdude: stk500_getsync() attempt 1 of 10: not in sync: resp=0x00",
    "avrdude: stk500_recv(): programmer is not responding",
    "error: expected ';' before '}}' token",
    "fatal error: {var}.h: No such file or directory",
    "exit status 1\nError compiling for board Arduino Uno.",
    "warning: unused variable '{var}'",
    "multiple definition of `loop`",
    "undefined reference to `setup`",
    "expected unqualified-id before '{var}'",
    
    # ESP32 Specific
    "A fatal error occurred: Failed to connect to ESP32: Timed out waiting for packet header",
    "Guru Meditation Error: Core  1 panic'ed (LoadProhibited). Exception was unhandled.",
    "Brownout detector was triggered",
    "rst:0x10 (RTCWDT_RTC_RESET),boot:0x13 (SPI_FAST_FLASH_BOOT)",
    "CORRUPT HEAP: Bad head at {var}",
    "E (123) spi_flash: flash read failed",
    "W (456) wifi: bss_handle is null",
    "Guru Meditation Error: Core  0 panic'ed (StoreProhibited)",
    "E (789) task_wdt: Task watchdog got triggered",
    "abort() was called at PC 0x40080000 on core 1"
]

LANGUAGES = ["Python", "Java", "C++", "JavaScript", "C#", "Ruby", "Go", "Rust", "Swift", "Kotlin", "PHP", "TypeScript", "Scala", "Dart", "Lua", "Haskell", "Perl", "R", "Objective-C", "Bash", "Arduino", "ESP32-C", "FreeRTOS"]

INSULTS = [
    "My gran could code better than this, and she's dead!",
    "You donkey! What is this absolute garbage?",
    "This code is so raw it's still running in the pasture!",
    "Are you having a laugh? This is an absolute disgrace!",
    "I've seen better logic in a bowl of alphabet soup!",
    "You call yourself a developer? This is a complete disaster!",
    "Shut it down! This memory leak is going to burn the whole server down!",
    "It's RAW! Your variables aren't even initialized, you idiot sandwich!",
    "This stack trace is longer than the menu at Kitchen Nightmares!",
    "Is this supposed to be an API? It looks like a dog's dinner!",
    "Wake up! You missed a semicolon and now the whole build is ruined!",
    "Look at this! Just look at it! It's pathetic!",
    "You've completely botched the syntax! Do it again!",
    "F*cking hell! This repo belongs in the bin!",
    "Get out! Get out of my IDE!",
    "This is the worst piece of code I have ever seen in my entire life!",
    "Even a monkey mashing the keyboard would write fewer bugs!",
    "What are you doing? Have you ever seen a tutorial in your life?",
    "This architecture is a complete joke! It's falling apart like a wet cake!",
    "You've butchered it! Absolutely butchered the framework!",
    "A culinary masterpiece? No, this is a code catastrophe!",
    "Where is the error handling? IT'S MISSING! WHERE IS IT?!",
    "Stop typing! You're making the compiler cry!",
    "This is so badly structured, a toddler with building blocks could do better!",
    "Do me a favor and delete your GitHub account. Now!",
    "I wouldn't trust this code to run a toaster!",
    "You've managed to write a bug that shouldn't even be mathematically possible!",
    "This is bland, boring, and fundamentally broken!",
    "If this codebase was a dish, I'd send it back to the kitchen and fire the chef!",
    "It's so bad, the garbage collector refuses to touch it!",
    "You call this a commit? I call it a cry for help!",
    "Are you trying to DDOS yourself? Because that's what this loop does!",
    "I'm allergic to this code. It's making me physically sick!",
    "Look at this indentation! Did you drop a plate of spaghetti on your keyboard?!",
    "This Arduino code couldn't even blink an LED without crashing the universe!",
    "Brownout triggered? More like brain-out triggered! You donkey!",
    "Guru Meditation Error? You need to meditate on why you chose this career path!",
    "Failed to connect to ESP32? It's trying to run away from your horrific code!"
]

TYPES = ["int", "str", "list", "dict", "NoneType", "User", "Config", "Object", "Array", "String", "Number", "Boolean", "uint8_t", "String", "TaskHandle_t", "QueueHandle_t", "SemaphoreHandle_t"]

def random_word(length=5):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

def generate_entry():
    lang = random.choice(LANGUAGES)
    template = random.choice(ERRORS)
    var_name = random_word(random.randint(3, 8))
    type_name = random.choice(TYPES)
    error = template.format(var=var_name, type=type_name)
    insult = random.choice(INSULTS)
    
    file_name = f"{random_word(random.randint(4, 10))}.{lang.lower()[:2]}"
    line_num = random.randint(1, 2000)
    
    prompt = f"<User>: IDE Error in {lang} at {file_name}:{line_num}:\n{error}"
    response = f"<Gordon>: {insult}"
    return f"{prompt}\n{response}"

def main():
    target_dir = "data/TinyStories_all_data"
    os.makedirs(target_dir, exist_ok=True)
    num_samples = 300000 # Increased number of samples
    num_shards = 20 # Splitting into more shards
    samples_per_shard = num_samples // num_shards
    
    # Shard 00 is for testing
    for shard_idx in range(num_shards):
        data = []
        for _ in range(samples_per_shard):
            data.append({"story": generate_entry()})
        
        file_path = os.path.join(target_dir, f"data{shard_idx:02d}.json")
        with open(file_path, "w") as f:
            json.dump(data, f)
        print(f"Generated {len(data)} error/insult pairs in {file_path}.")

if __name__ == "__main__":
    main()
